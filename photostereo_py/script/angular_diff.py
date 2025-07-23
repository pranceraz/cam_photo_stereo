# ---------------------------------------------------------------
# angular_diff_fixed.py
# ---------------------------------------------------------------
import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple
from segment_anything import SamPredictor, sam_model_registry
# ---------------------------------------------------------------
# 1.  I/O helpers
# ---------------------------------------------------------------

def load_16bit_normal(path: str) -> np.ndarray:
    """
    Load a 16-bit RGB normal-map PNG/TIFF and return unit vectors in [-1,1].
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.dtype != np.uint16 or img.shape[2] != 3:
        raise ValueError(f"Invalid 16-bit RGB image: {path}")

    normals = img.astype(np.float32) / 65535.0        # [0,1]
    normals = normals * 2.0 - 1.0                     # [-1,1]
    normals[..., 2] = np.abs(normals[..., 2])         # Z⁺ hemisphere
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= np.maximum(norm, 1e-6)
    return normals


# ---------------------------------------------------------------
# 2.  Cosine-similarity (GPU, patch-wise)
# ---------------------------------------------------------------

def sliding_cosine_similarity_gpu(
    ref: torch.Tensor, test: torch.Tensor, patch_size: int = 31
) -> torch.Tensor:
    """
    Patch-wise cosine similarity (mean over each patch).
    ref, test : shape (1,3,H,W) on the same CUDA device.
    Returns   : (H,W) tensor on the same device.
    """
    device = ref.device
    _, _, H, W = ref.shape
    pad = patch_size // 2

    ref_pad  = F.pad(ref,  (pad, pad, pad, pad), mode='reflect')
    test_pad = F.pad(test, (pad, pad, pad, pad), mode='reflect')

    simil = torch.empty((H, W), dtype=torch.float32, device=device)

    # process row-chunks to save memory
    chunk = 64
    for y0 in range(0, H, chunk):
        y1 = min(y0 + chunk, H)
        ch = y1 - y0

        ref_patch  = torch.zeros((ch, W, patch_size, patch_size, 3),
                                 device=device)
        test_patch = torch.zeros_like(ref_patch)

        for i, y in enumerate(range(y0, y1)):
            for x in range(W):
                rp = ref_pad[0, :, y:y+patch_size, x:x+patch_size].permute(1,2,0)
                tp = test_pad[0,:, y:y+patch_size, x:x+patch_size].permute(1,2,0)
                ref_patch[i, x]  = rp
                test_patch[i, x] = tp

        ref_f  = ref_patch.reshape(-1, patch_size*patch_size, 3)
        test_f = test_patch.reshape_as(ref_f)
        dots   = (ref_f * test_f).sum(dim=2)           # (N,P)
        mean_d = dots.mean(dim=1).view(ch, W)          # (ch,W)
        simil[y0:y1] = mean_d

        del ref_patch, test_patch, ref_f, test_f, dots, mean_d
        torch.cuda.empty_cache()
        print(f"[INFO] Processed rows {y0+1}–{y1}/{H}")

    return simil


# ---------------------------------------------------------------
# 3.  CLAHE enhancement (CPU)
# ---------------------------------------------------------------

def apply_clahe_enhancement_gpu(
    sim_map: torch.Tensor, clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> torch.Tensor:
    sim_cpu = sim_map.cpu().numpy()
    sim_8u  = ((sim_cpu + 1.0) / 2.0 * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enh_8u = clahe.apply(sim_8u)

    enh = enh_8u.astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(enh).to(sim_map.device)


# ---------------------------------------------------------------
# 4.  Save utilities
# ---------------------------------------------------------------

def save_similarity_maps(sim: torch.Tensor, enh: torch.Tensor, stem: str) -> None:
    sim_8u = ((sim.cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
    enh_8u = ((enh.cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)

    sim_p = f"{os.path.splitext(stem)[0]}_similarity.png"
    enh_p = f"{os.path.splitext(stem)[0]}_enhanced.png"

    cv2.imwrite(sim_p, sim_8u)
    cv2.imwrite(enh_p, enh_8u)
    print(f"[INFO] Similarity map  → {sim_p}")
    print(f"[INFO] Enhanced map    → {enh_p}")


# ---------------------------------------------------------------
# 5.  Main similarity pipeline
# ---------------------------------------------------------------

def run_gpu_similarity_analysis(
    ref_path: str, test_path: str, output_base: str,
    patch_size: int = 31, clip_limit: float = 3.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    ref_np = load_16bit_normal(ref_path)
    test_np = load_16bit_normal(test_path)
    if ref_np.shape != test_np.shape:
        raise ValueError("Normal-map shapes differ.")

    ref_t = torch.from_numpy(ref_np).permute(2,0,1).unsqueeze(0).to(device)
    test_t= torch.from_numpy(test_np).permute(2,0,1).unsqueeze(0).to(device)

    sim_t = sliding_cosine_similarity_gpu(ref_t, test_t, patch_size)
    enh_t = apply_clahe_enhancement_gpu(sim_t, clip_limit, tile_grid_size)

    save_similarity_maps(sim_t, enh_t, output_base)
    return sim_t, enh_t


# ---------------------------------------------------------------
# 6.  SAM segmentation on the heat-map
# ---------------------------------------------------------------

def sam_from_similarity(
    sim_png_path: str,
    sam_ckpt: str = r"C:/Users/Photogauge/projet/cam_photo_stereo/sam_vit_b.pth",
    peak_frac: float = 0.001,
    dilate_px: int = 4
) -> np.ndarray:
    """
    Segment scratches directly on the enhanced similarity PNG.
    Returns a uint8 mask (1 = scratch).
    """
    sim = cv2.imread(sim_png_path, cv2.IMREAD_GRAYSCALE)
    if sim is None:
        raise FileNotFoundError(sim_png_path)

    # SAM expects 3-channel input → stack the heat-map
    sim_rgb = cv2.merge([sim, sim, sim])

    sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt).to("cuda")
    predictor = SamPredictor(sam)
    predictor.set_image(sim_rgb)

    # auto-prompt at brightest region
    thresh = np.percentile(sim, 100 * (1 - peak_frac))
    seeds  = np.column_stack(np.nonzero(sim >= thresh))
    if seeds.size == 0:                              # fallback
        y, x = np.unravel_index(sim.argmax(), sim.shape)
    else:
        y, x = seeds.mean(axis=0).astype(int)

    masks, _, _ = predictor.predict(
        point_coords=np.array([[x, y]]),
        point_labels=np.array([1]),
        multimask_output=False
    )
    mask = masks[0].astype(np.uint8)

    # optional clean-up
    if dilate_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (dilate_px*2+1, dilate_px*2+1))
        mask = cv2.dilate(mask, k, 1)

    cv2.imwrite("scratch_mask.png", mask*255)
    print("[INFO] scratch_mask.png written")
    return mask


# ---------------------------------------------------------------
# 7.  Overlay + polygon export (visualisation)
# ---------------------------------------------------------------

def overlay_and_polygons(
    sim_png: str, mask: np.ndarray,
    overlay_png: str = "scratch_overlay.png",
    polygons_json: str = "scratch_polygons.json",
    color: tuple[int,int,int] = (0,255,0)
):
    base = cv2.imread(sim_png, cv2.IMREAD_GRAYSCALE)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    base_rgb[mask == 1] = color
    cv2.imwrite(overlay_png, base_rgb)
    print(f"[INFO] overlay saved  → {overlay_png}")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = [c.squeeze(1).tolist() for c in contours if c.size >= 6]
    with open(polygons_json, "w") as f:
        json.dump({"polygons": polys, "h": mask.shape[0], "w": mask.shape[1]}, f, indent=2)
    print(f"[INFO] polygons saved → {polygons_json}")

def pop_contrast(
    in_png: str,
    out_png: str = "pop_similarity.png",
    gamma: float = 0.35,
    keep_top: float = 0.002,
    morph_kernel: int = 5,
    morph_iter: int = 2
) -> np.ndarray:
    """
    Ultra-boost the contrast of an 8-bit similarity map so scratches
    become bright and the background fades.

    Parameters
    ----------
    in_png      : path to the 8-bit similarity PNG (raw or CLAHE-enhanced).
    out_png     : file to write the pop-contrast image.
    gamma       : <1 brightens highlights, darkens mid-tones (0.25–0.5 typical).
    keep_top    : fraction of hottest pixels kept after thresholding (0–1).
    morph_kernel: diameter (px) of the elliptical kernel for morphological close.
    morph_iter  : iterations of morphology to run.

    Returns
    -------
    binary_mask : np.ndarray uint8, shape(H,W), 1 where scratch pixels remain.
    """
    # 1 ── load the single-channel similarity map
    sim = cv2.imread(in_png, cv2.IMREAD_GRAYSCALE)
    if sim is None:
        raise FileNotFoundError(in_png)

    # 2 ── percentile clip (1–99 %) to discard outliers, then stretch to 0-255
    lo, hi = np.percentile(sim, [1, 99])
    sim_clipped = np.clip(sim, lo, hi)
    stretched = ((sim_clipped - lo) * (255.0 / (hi - lo))).astype(np.uint8)

    # 3 ── gamma correction <1 → boosts highlights
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)],
                   dtype="uint8")
    popped = cv2.LUT(stretched, lut)

    # 4 ── keep only the top X % brightest pixels
    thresh_val = np.percentile(popped, 100 * (1 - keep_top))
    binary = (popped >= thresh_val).astype(np.uint8)

    # 5 ── morphology: close gaps, thicken hair-line scratches
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              kernel, iterations=morph_iter)

    # 6 ── write the high-contrast map for visual confirmation
    cv2.imwrite(out_png, popped)
    print(f"[INFO] pop-contrast map  → {out_png}")
    return binary


# ---------------------------------------------------------------
# 8.  Example usage
# ---------------------------------------------------------------
if __name__ == "__main__":
    ref_path  = r"C:/Users/Photogauge/projet/cam_photo_stereo/photostereo_py/script/normal_map_50paise_bef.png"
    test_path = r"C:/Users/Photogauge/projet/cam_photo_stereo/photostereo_py/script/after.png"
    output_base = "similarity_analysis.png"

    # #--- step-1: similarity analysis (uncomment to recompute) -----------
    # run_gpu_similarity_analysis(
    #     ref_path, test_path, output_base,
    #     patch_size=31, clip_limit=3.0, tile_grid_size=(8,8)
    # )

    # --- step-2: SAM segmentation on the enhanced map -------------------
    sim_png = "similarity_analysis_similarity.png"     # <- produced in step-1
    binary_mask = pop_contrast("similarity_analysis_enhanced.png")
    mask = sam_from_similarity("pop_similarity.png")

    # --- step-3: visual overlay & polygons ------------------------------
    overlay_and_polygons(sim_png, mask)
