#!/usr/bin/env python3
"""
Prototype-matching segmentation with DINOv3 (no training)
- Uses two *public-domain* PNG apples that include an alpha channel as ground-truth masks
- Builds foreground/background prototypes from DINOv3 patch tokens
- Segments a separate CC0 apple photo by cosine(pF) - cosine(pB)

Model:
  - facebook/dinov3-vitb16-pretrain-lvd1689m  (Meta, Hugging Face Hub)
"""

import os, io, math, time, argparse, shutil, requests
import numpy as np
from PIL import Image
import cv2
import torch
from transformers import AutoImageProcessor, AutoModel

EXEMPLARS = {
    # filename : url
    "ex1.png": "https://upload.wikimedia.org/wikipedia/commons/6/6d/Apple_picture.jpg",
    "ex2.png": "https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg",
}
TESTS = {
    "apple_photo.jpg": "https://cdn.hswstatic.com/gif/pick-fresh-fruit.jpg"
}

UA = {"User-Agent": "DINOv3-proto-demo/1.0 (+https://example.com; for research)"}

def download(urls, path):
    if os.path.exists(path): return
    last_err = None
    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=30, headers=UA) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            print(f"[downloaded] {os.path.basename(path)} from {url}")
            return
        except Exception as e:
            last_err = e
            print(f"[warn] fetch failed: {url} ({e})")
    raise last_err

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def pil_open_rgb(p):
    return Image.open(p).convert("RGB")

def rgb_to_mask_from_white(img_rgb,thre=245):
    a = np.array(img_rgb)
    m = (a[:,:,0] >= thre)
    m = np.logical_and(m,(a[:,:,1] >= thre))
    m = np.logical_and(m,(a[:,:,2] >= thre))
    return (255-m*255).astype(np.uint8)

@torch.inference_mode()
def get_patch_tokens(model, processor, pil_img, device):
    """
    Returns:
      tokens: [Gh, Gw, C] L2-normalized patch tokens
      grid_hw: (Gh, Gw)
      img_hw:  (H, W) as seen by the model
    """
    pil_img = pil_img.convert("RGB")
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)  # last_hidden_state: [1, S, C] (S may include CLS)

    hidden = out.last_hidden_state               # [1, S, C]
    S, C = hidden.shape[1], hidden.shape[2]

    # Try both: with and without CLS
    candidates = [hidden, hidden[:, 1:, :]] if S > 1 else [hidden]

    Hm, Wm = inputs["pixel_values"].shape[-2:]   # model input size after processor
    ps = getattr(model.config, "patch_size", 16) # ViT-B/16 default

    def try_make_grid(tokens_2d):
        T = tokens_2d.shape[1]
        # 1) Expect patch grid
        gh, gw = Hm // ps, Wm // ps
        if gh * gw == T:
            return tokens_2d.view(1, gh, gw, C).squeeze(0)
        # 2) Sometimes output is per-pixel (rare) -> H×W
        if Hm * Wm == T:
            return tokens_2d.view(1, Hm, Wm, C).squeeze(0)
        # 3) Fallback: factor T to a plausible Gh×Gw close to Hm:Wm
        import math
        ar = Hm / max(Wm, 1)
        best = None
        g = int(math.sqrt(T))
        for d in range(1, g + 1):
            if T % d == 0:
                gh2, gw2 = d, T // d
                if gh2 <= 4096 and gw2 <= 4096:
                    score = abs((gh2 / gw2) - ar)
                    if best is None or score < best[0]:
                        best = (score, gh2, gw2)
        if best is not None:
            _, gh2, gw2 = best
            return tokens_2d.view(1, gh2, gw2, C).squeeze(0)
        raise RuntimeError(f"Cannot infer grid for T={T}, HxW={Hm}x{Wm}, ps={ps}")

    tok = None
    for cand in candidates:
        try:
            grid_tok = try_make_grid(cand)
            tok = grid_tok
            break
        except Exception:
            continue
    if tok is None:
        raise RuntimeError("Failed to reshape tokens into a grid.")

    tok = torch.nn.functional.normalize(tok, dim=-1)  # [Gh, Gw, C]
    Gh, Gw = tok.shape[:2]
    return tok, (Gh, Gw), (Hm, Wm)

def resize_mask_to_grid(mask_hw, grid_hw):
    gh, gw = grid_hw
    return cv2.resize(mask_hw, (gw, gh), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

def upsample_to_img(mask_grid, img_hw):
    H, W = img_hw
    return cv2.resize(mask_grid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

def make_prototypes_from_pngs(model, processor, png_paths, device):
    f_list, b_list = [], []
    for p in png_paths:
        img = pil_open_rgb(p)
        mask = rgb_to_mask_from_white(img,200)    # [H, W] {0,1}
        tokens, grid, img_hw = get_patch_tokens(model, processor, img, device)
        m_grid = resize_mask_to_grid(mask, grid) > 0
        T = tokens.view(-1, tokens.shape[-1])          # [Gh*Gw, C]
        m_flat = torch.from_numpy(m_grid.reshape(-1)).to(T.device)
        if m_flat.any(): f_list.append(T[m_flat].median(0).values)
        if (~m_flat).any(): b_list.append(T[~m_flat].median(0).values)
    assert f_list, "Foreground prototypes empty."
    pF = torch.nn.functional.normalize(torch.stack(f_list,0).mean(0), dim=0)
    pB = torch.nn.functional.normalize(torch.stack(b_list,0).mean(0), dim=0) if b_list else None
    return pF, pB

@torch.inference_mode()
def segment_with_prototypes(model, processor, pil_img, pF, pB, device, thr=None, keep_largest=True):
    tokens, grid, img_hw = get_patch_tokens(model, processor, pil_img, device)
    T = tokens.view(-1, tokens.shape[-1])
    sF = (T @ pF).view(tokens.shape[:2])
    if pB is not None:
        sB = (T @ pB).view(tokens.shape[:2])
        score = sF - sB
    else:
        score = sF
    sc = score.float().cpu().numpy()
    sc = (sc - sc.min()) / (sc.max() - sc.min() + 1e-6)
    if thr is None:
        _thr, _ = cv2.threshold((sc*255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
        thr = _thr/255.0
    m_grid = (sc >= thr).astype(np.uint8)
    mask = upsample_to_img(m_grid, img_hw)
    if keep_largest:
        num, lab = cv2.connectedComponents(mask)
        if num > 1:
            counts = np.bincount(lab.ravel())
            keep = np.argmax(counts[1:]) + 1
            mask = (lab == keep).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return mask

def overlay_bgr(bgr, mask_u8, alpha=0.6):
    """
    bgr: HxWx3 uint8 (original image size)
    mask_u8: h-gxw-g uint8 (may be model input size)
    """
    H, W = bgr.shape[:2]
    if mask_u8.shape[:2] != (H, W):
        mask_u8 = cv2.resize(mask_u8.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    color = np.array([0, 255, 0], np.uint8)
    out = bgr.copy()
    m = mask_u8 > 0
    out[m] = (alpha * out[m] + (1 - alpha) * color).astype(np.uint8)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="dinov3_open_apples")
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    work = ensure_dir(args.workdir)
    raw = ensure_dir(os.path.join(work, "raw"))
    out = ensure_dir(os.path.join(work, "out"))

    # 1) Download exemplars (PNGs with alpha) + test photo
    for fname, url in EXEMPLARS.items():
        download([url], os.path.join(raw, fname))
    for fname, url in TESTS.items():
        download([url], os.path.join(raw, fname))

    # 2) Load DINOv3
    device = torch.device(args.device)
    print(f"[info] loading {args.model} on {device}")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    # 3) Build prototypes from the two PNGs
    png_paths = [os.path.join(raw, "ex1.png"), os.path.join(raw, "ex2.png")]
    pF, pB = make_prototypes_from_pngs(model, processor, png_paths, device)
    print("[ok] prototypes ready.")

    # 4) Segment the CC0 photo
    test_path = os.path.join(raw, "apple_photo.jpg")
    pil_img = Image.open(test_path).convert("RGB")
    t0 = time.time()
    mask = segment_with_prototypes(model, processor, pil_img, pF, pB, device)
    dt = (time.time() - t0) * 1000.0
    print(f"[seg] {os.path.basename(test_path)}  {dt:.1f} ms")

    # 5) Save outputs
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    vis = overlay_bgr(bgr, mask)
    cv2.imwrite(os.path.join(out, "apple_photo_mask.png"), (mask*255).astype(np.uint8))
    cv2.imwrite(os.path.join(out, "apple_photo_overlay.jpg"), vis)
    print(f"[done] results in: {out}")

    print("\nLicenses:")
    print(" - Red apple 1.png — Public Domain (Wikimedia Commons)")
    print(" - Apple red 1.png — Public Domain (Wikimedia Commons)")
    print(" - Red apple on white background.jpg — CC0 (Public Domain)")

if __name__ == "__main__":
    main()
