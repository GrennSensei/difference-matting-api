import io
import gc
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from PIL import Image, ImageFilter
import numpy as np
from collections import deque

app = FastAPI(title="Difference Matting API", version="2.1.0")


# =========================================================
# Utilities
# =========================================================

def load_upload_as_rgb(file: UploadFile) -> Image.Image:
    data = file.file.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def resize_max(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def pil_rgb_to_f16(img: Image.Image) -> np.ndarray:
    # [H,W,3] float16 (RAM friendly)
    return np.array(img, dtype=np.uint8).astype(np.float16)


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def alpha_to_u8(alpha01: np.ndarray) -> np.ndarray:
    # alpha01 float -> uint8
    return (clamp01(alpha01) * 255.0).round().astype(np.uint8)


def u8_to_alpha01(a: np.ndarray) -> np.ndarray:
    # uint8 -> float16 alpha 0..1
    return (a.astype(np.float16) / np.float16(255.0))


def edge_only_soften(alpha_u8: np.ndarray, sigma: float, low: int, high: int) -> np.ndarray:
    """
    Blur ONLY the edge band: low<alpha<high.
    Prevents spreading background noise.
    """
    if sigma <= 0:
        return alpha_u8

    a_img = Image.fromarray(alpha_u8, mode="L")
    blurred = a_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    blurred_u8 = np.array(blurred, dtype=np.uint8)

    edge = (alpha_u8 > low) & (alpha_u8 < high)
    out = alpha_u8.copy()
    out[edge] = blurred_u8[edge]
    return out


def dilate_mask(mask_bool: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Fast dilation using PIL MaxFilter on an 8-bit mask.
    """
    if radius_px <= 0:
        return mask_bool
    k = radius_px * 2 + 1
    m = (mask_bool.astype(np.uint8) * 255)
    m_img = Image.fromarray(m, mode="L")
    d_img = m_img.filter(ImageFilter.MaxFilter(size=k))
    d = np.array(d_img, dtype=np.uint8) > 0
    return d


def remove_small_specks(alpha_u8: np.ndarray, candidate_mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove small connected components in candidate_mask by setting alpha=0 there.
    candidate_mask: bool [H,W] where pixels are eligible to be removed.
    """
    if min_area <= 0:
        return alpha_u8

    h, w = alpha_u8.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    out = alpha_u8.copy()

    def neighbors(y, x):
        if y > 0: yield y - 1, x
        if y < h - 1: yield y + 1, x
        if x > 0: yield y, x - 1
        if x < w - 1: yield y, x + 1

    # Small optimization: iterate only where candidate is true
    ys, xs = np.where(candidate_mask)
    for idx in range(len(ys)):
        y0 = int(ys[idx])
        x0 = int(xs[idx])
        if visited[y0, x0]:
            continue

        q = deque([(y0, x0)])
        visited[y0, x0] = 1
        comp = [(y0, x0)]

        while q:
            y, x = q.popleft()
            for ny, nx in neighbors(y, x):
                if visited[ny, nx]:
                    continue
                if not candidate_mask[ny, nx]:
                    continue
                visited[ny, nx] = 1
                q.append((ny, nx))
                comp.append((ny, nx))

        if len(comp) < min_area:
            for (yy, xx) in comp:
                out[yy, xx] = 0

    return out


def fill_small_holes(alpha_u8: np.ndarray, min_area: int) -> np.ndarray:
    """
    Fill small enclosed transparent holes in alpha.
    """
    if min_area <= 0:
        return alpha_u8

    h, w = alpha_u8.shape
    visited = np.zeros((h, w), dtype=np.uint8)

    def neighbors(y, x):
        if y > 0: yield y-1, x
        if y < h-1: yield y+1, x
        if x > 0: yield y, x-1
        if x < w-1: yield y, x+1

    q = deque()
    for x in range(w):
        if alpha_u8[0, x] == 0: q.append((0, x))
        if alpha_u8[h-1, x] == 0: q.append((h-1, x))
    for y in range(h):
        if alpha_u8[y, 0] == 0: q.append((y, 0))
        if alpha_u8[y, w-1] == 0: q.append((y, w-1))

    while q:
        y, x = q.popleft()
        if visited[y, x]:
            continue
        if alpha_u8[y, x] != 0:
            continue
        visited[y, x] = 1
        for ny, nx in neighbors(y, x):
            if not visited[ny, nx] and alpha_u8[ny, nx] == 0:
                q.append((ny, nx))

    hole_vis = np.zeros((h, w), dtype=np.uint8)
    out = alpha_u8.copy()

    for y0 in range(h):
        for x0 in range(w):
            if out[y0, x0] != 0:
                continue
            if visited[y0, x0] == 1:
                continue
            if hole_vis[y0, x0] == 1:
                continue

            comp = []
            qq = deque([(y0, x0)])
            hole_vis[y0, x0] = 1

            while qq:
                y, x = qq.popleft()
                comp.append((y, x))
                for ny, nx in neighbors(y, x):
                    if out[ny, nx] == 0 and visited[ny, nx] == 0 and hole_vis[ny, nx] == 0:
                        hole_vis[ny, nx] = 1
                        qq.append((ny, nx))

            if len(comp) <= min_area:
                for (yy, xx) in comp:
                    out[yy, xx] = 255

    return out


def build_bg_map_lowres(img: Image.Image, blur_radius: float, bg_map_side: int) -> Image.Image:
    """
    Memory-safe background model:
    - downscale to bg_map_side
    - blur there
    - upscale back to original
    """
    if blur_radius <= 0:
        return img

    w, h = img.size
    if bg_map_side <= 0:
        bg_map_side = 512

    # Downscale
    small = resize_max(img, max_side=bg_map_side)
    # Blur on low-res
    small_blur = small.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    # Upscale back
    return small_blur.resize((w, h), Image.BILINEAR)


# =========================================================
# Difference Matting Endpoint (Pro-stabilized, RAM-safe)
# =========================================================

@app.post("/matte/difference")
async def difference_matte(
    white_file: UploadFile = File(..., description="Artwork on (near) white background"),
    black_file: UploadFile = File(..., description="Artwork on (near) black background"),

    # size / perf (Render Free safe defaults)
    max_side: int = Query(2048, ge=512, le=6000),

    # background modeling (low-res map)
    bg_blur: float = Query(18.0, ge=0.0, le=64.0),
    bg_map_side: int = Query(512, ge=128, le=1024),

    # alpha cleanup (kills random bg specks)
    noise_cut: float = Query(0.025, ge=0.0, le=0.2),           # slightly higher default for dust
    fg_thresh: float = Query(0.25, ge=0.0, le=1.0),
    keep_near_fg_px: int = Query(4, ge=0, le=32),
    speck_alpha_max: float = Query(0.25, ge=0.0, le=1.0),
    min_speck_area: int = Query(150, ge=0, le=50000),
    speck_min_pixels_to_run: int = Query(2000, ge=0, le=20000000),  # skip BFS if too few candidates

    # edge soften (edge-only, won't spread bg noise)
    edge_soften: int = Query(1, ge=0, le=1),
    soften_sigma: float = Query(0.9, ge=0.0, le=10.0),
    edge_low: int = Query(10, ge=0, le=254),
    edge_high: int = Query(245, ge=1, le=255),

    # hole fill (optional)
    hole_remove: int = Query(0, ge=0, le=1),
    min_hole_area: int = Query(800, ge=0, le=500000),
):
    try:
        # 1) Load + resize
        img_w = resize_max(load_upload_as_rgb(white_file), max_side=max_side)
        img_b = resize_max(load_upload_as_rgb(black_file), max_side=max_side)

        if img_w.size != img_b.size:
            return JSONResponse(
                status_code=400,
                content={"error": "Dimension mismatch: White and black images must have identical size."}
            )

        # 2) Convert to arrays (float16 = RAM friendly)
        W = pil_rgb_to_f16(img_w)  # [H,W,3]
        B = pil_rgb_to_f16(img_b)

        # 3) Build per-pixel background maps using LOW-RES blur (RAM safe)
        BgW_img = build_bg_map_lowres(img_w, blur_radius=bg_blur, bg_map_side=bg_map_side)
        BgB_img = build_bg_map_lowres(img_b, blur_radius=bg_blur, bg_map_side=bg_map_side)
        BgW = pil_rgb_to_f16(BgW_img)
        BgB = pil_rgb_to_f16(BgB_img)

        # 4) Robust alpha (per-channel -> median), with safe denom
        denom = (BgW - BgB)  # float16
        # clamp denom magnitude to >= 1 to avoid exploding alpha
        denom_safe = np.where(np.abs(denom) < 1.0, np.sign(denom) * 1.0, denom).astype(np.float16)

        alpha_rgb = 1.0 - ((W - B) / denom_safe)   # float16
        alpha = np.median(alpha_rgb, axis=2)       # float16
        alpha = clamp01(alpha).astype(np.float16)

        # free some big temps early
        del alpha_rgb, denom, denom_safe
        gc.collect()

        # 5) Hard noise cut (kills faint confetti)
        if noise_cut > 0:
            alpha = np.where(alpha < np.float16(noise_cut), np.float16(0.0), alpha).astype(np.float16)

        # 6) Foreground protection region
        fg = alpha >= np.float16(fg_thresh)
        fg_near = dilate_mask(fg, radius_px=keep_near_fg_px)

        # 7) Speck removal (only if enough candidates to justify BFS)
        cand = (alpha > np.float16(0.0)) & (alpha < np.float16(speck_alpha_max)) & (~fg_near)

        alpha_u8 = alpha_to_u8(alpha)

        cand_count = int(np.count_nonzero(cand))
        if min_speck_area > 0 and cand_count >= speck_min_pixels_to_run:
            alpha_u8 = remove_small_specks(alpha_u8, candidate_mask=cand, min_area=min_speck_area)

        # Release masks early
        del cand, fg, fg_near
        gc.collect()

        alpha = u8_to_alpha01(alpha_u8)  # float16

        # 8) Optional hole fill (inside object)
        if hole_remove == 1 and min_hole_area > 0:
            alpha_u8 = alpha_to_u8(alpha)
            alpha_u8 = fill_small_holes(alpha_u8, min_area=min_hole_area)
            alpha = u8_to_alpha01(alpha_u8)

        # 9) Edge-only soften
        alpha_u8 = alpha_to_u8(alpha)
        if edge_soften == 1 and soften_sigma > 0:
            alpha_u8 = edge_only_soften(alpha_u8, sigma=soften_sigma, low=edge_low, high=edge_high)
            alpha = u8_to_alpha01(alpha_u8)

        # 10) Color recovery (background-aware), avoid recovering dust
        # C = (B - (1-a)*BgB) / a
        a = alpha  # float16
        a_safe = np.maximum(a, np.float16(1e-3))[..., None]  # float16
        out_rgb = (B - (np.float16(1.0) - a)[..., None] * BgB) / a_safe
        out_rgb = np.clip(out_rgb, 0.0, 255.0)

        # zero RGB where alpha is 0-ish to avoid colored dust
        out_rgb = np.where((a[..., None] <= np.float16(noise_cut)), 0.0, out_rgb)
        out_rgb_u8 = out_rgb.astype(np.uint8)

        out_a_u8 = alpha_to_u8(a)

        # free big arrays before PIL compose (reduce peak)
        del W, B, BgW, BgB, out_rgb, a, a_safe, alpha
        gc.collect()

        # 11) Build RGBA
        out = Image.fromarray(out_rgb_u8, mode="RGB").convert("RGBA")
        out.putalpha(Image.fromarray(out_a_u8, mode="L"))

        return StreamingResponse(io.BytesIO(png_bytes(out)), media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {e}"})


# =========================================================
# Web test page
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Difference Matting Test</title>
  </head>
  <body style="font-family: Arial; max-width: 980px; margin: 40px auto; line-height:1.45;">
    <h2>Difference Matting Test (Pro + RAM-safe)</h2>
    <p>Upload a <b>white</b> and a <b>black</b> background version of the same artwork (pixel-aligned).</p>

    <form action="/matte/difference?max_side=2048&bg_blur=18&bg_map_side=512&noise_cut=0.025&fg_thresh=0.25&keep_near_fg_px=4&speck_alpha_max=0.25&min_speck_area=150&speck_min_pixels_to_run=2000&edge_soften=1&soften_sigma=0.9&hole_remove=0&min_hole_area=800"
          method="post" enctype="multipart/form-data"
          style="padding:16px; border:1px solid #ddd; border-radius:10px;">
      <p><b>White image:</b></p>
      <input type="file" name="white_file" accept="image/*" required />

      <p style="margin-top:14px;"><b>Black image:</b></p>
      <input type="file" name="black_file" accept="image/*" required />

      <p style="margin-top:14px;">
        <button type="submit" style="padding:10px 14px; font-weight:bold; cursor:pointer;">
          Generate Transparent PNG
        </button>
      </p>
    </form>

    <hr style="margin: 22px 0; border: none; border-top: 1px solid #eee;" />

    <h3 style="margin-bottom:6px;">Recommended knobs</h3>
    <ul style="margin-top:6px; color:#444;">
      <li><b>max_side</b>: Render Free-hez 1600–2048 ajánlott. 3000 már könnyen OOM.</li>
      <li><b>noise_cut</b> (0.02–0.05): magasabb = kevesebb “confetti”, de puha élek sérülhetnek.</li>
      <li><b>min_speck_area</b> (80–300): magasabb = agresszívebb pötty-eltávolítás.</li>
      <li><b>bg_blur</b> (12–24) + <b>bg_map_side</b> (384–768): gradient ellen, de RAM-safe.</li>
    </ul>

    <p style="color:#777;">
      Ha még mindig látszik pöttyözés: emeld <b>noise_cut</b>-ot 0.03-ra vagy <b>min_speck_area</b>-t 200-ra.
      Ha túl “kivágott”: csökkentsd <b>noise_cut</b>-ot 0.02-re.
    </p>
  </body>
</html>
"""

@app.get("/health")
async def health():
    return {"status": "ok", "endpoint": "/matte/difference"}
