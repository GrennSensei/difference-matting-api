import io
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from PIL import Image, ImageFilter
import numpy as np
from collections import deque

app = FastAPI(title="Difference Matting API", version="2.0.0")

# =========================================================
# Utilities
# =========================================================

def load_upload_as_rgb(file: UploadFile) -> Image.Image:
    data = file.file.read()
    # NOTE: EXIF rotáció néha gond — de most maradunk minimalon.
    return Image.open(io.BytesIO(data)).convert("RGB")

def resize_max(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), Image.LANCZOS)

def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def pil_rgb_to_f32(img: Image.Image) -> np.ndarray:
    # [H,W,3] float32
    return np.array(img, dtype=np.uint8).astype(np.float32)

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def alpha_to_u8(alpha01: np.ndarray) -> np.ndarray:
    return (clamp01(alpha01) * 255.0).round().astype(np.uint8)

def u8_to_alpha01(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) / 255.0

def gaussian_blur_rgb(img: Image.Image, radius: float) -> Image.Image:
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

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
    radius_px=0 -> no change.
    """
    if radius_px <= 0:
        return mask_bool
    k = radius_px * 2 + 1
    m = (mask_bool.astype(np.uint8) * 255)
    m_img = Image.fromarray(m, mode="L")
    # MaxFilter does dilation on grayscale masks
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

    for y0 in range(h):
        row_cand = candidate_mask[y0]
        row_vis = visited[y0]
        for x0 in range(w):
            if not row_cand[x0]:
                continue
            if row_vis[x0]:
                continue

            # BFS component
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
    Fill small enclosed transparent holes in alpha (same as your original).
    alpha_u8: uint8 [H,W]
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

    # mark border-connected transparent pixels
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

    # find enclosed holes and fill if small
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


# =========================================================
# Difference Matting Endpoint (Pro-stabilized)
# =========================================================

@app.post("/matte/difference")
async def difference_matte(
    white_file: UploadFile = File(..., description="Artwork on (near) white background"),
    black_file: UploadFile = File(..., description="Artwork on (near) black background"),

    # size / perf
    max_side: int = Query(3000, ge=512, le=6000),

    # background modeling (handles non-pure bg + gradients)
    bg_blur: float = Query(24.0, ge=0.0, le=64.0),

    # alpha cleanup (kills random bg specks)
    noise_cut: float = Query(0.02, ge=0.0, le=0.2),              # alpha below this -> 0
    fg_thresh: float = Query(0.25, ge=0.0, le=1.0),              # "confident foreground"
    keep_near_fg_px: int = Query(4, ge=0, le=32),                # protect pixels near fg from being removed
    speck_alpha_max: float = Query(0.25, ge=0.0, le=1.0),        # speck candidates are below this alpha
    min_speck_area: int = Query(150, ge=0, le=50000),            # remove small specks under this area

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

        # 2) Convert to arrays
        W = pil_rgb_to_f32(img_w)  # [H,W,3]
        B = pil_rgb_to_f32(img_b)

        # 3) Build per-pixel background maps (handles off-white/off-black + gradients)
        BgW_img = gaussian_blur_rgb(img_w, radius=bg_blur)
        BgB_img = gaussian_blur_rgb(img_b, radius=bg_blur)
        BgW = pil_rgb_to_f32(BgW_img)
        BgB = pil_rgb_to_f32(BgB_img)

        # 4) Robust alpha (per-channel -> median)
        # alpha_c = 1 - (W-B)/(BgW-BgB)
        denom = (BgW - BgB)
        # Avoid division explosions where denom is tiny
        denom_safe = np.where(np.abs(denom) < 1.0, np.sign(denom) * 1.0, denom)  # clamp denom magnitude to >=1

        alpha_rgb = 1.0 - ((W - B) / denom_safe)  # [H,W,3]
        alpha = np.median(alpha_rgb, axis=2)       # [H,W]
        alpha = clamp01(alpha)

        # 5) Hard noise cut (kills faint confetti)
        if noise_cut > 0:
            alpha = np.where(alpha < noise_cut, 0.0, alpha)

        # 6) Foreground protection region (don’t delete near true fg edges)
        fg = alpha >= fg_thresh
        fg_near = dilate_mask(fg, radius_px=keep_near_fg_px)

        # 7) Speck removal (small low-alpha components away from fg)
        # Candidate specks: low alpha, not near fg
        cand = (alpha > 0.0) & (alpha < speck_alpha_max) & (~fg_near)
        alpha_u8 = alpha_to_u8(alpha)

        if min_speck_area > 0 and np.any(cand):
            alpha_u8 = remove_small_specks(alpha_u8, candidate_mask=cand, min_area=min_speck_area)
            alpha = u8_to_alpha01(alpha_u8)

        # 8) Optional hole fill (inside object)
        if hole_remove == 1 and min_hole_area > 0:
            alpha_u8 = alpha_to_u8(alpha)
            alpha_u8 = fill_small_holes(alpha_u8, min_area=min_hole_area)
            alpha = u8_to_alpha01(alpha_u8)

        # 9) Edge-only soften (won’t smear background noise)
        alpha_u8 = alpha_to_u8(alpha)
        if edge_soften == 1 and soften_sigma > 0:
            alpha_u8 = edge_only_soften(alpha_u8, sigma=soften_sigma, low=edge_low, high=edge_high)
            alpha = u8_to_alpha01(alpha_u8)

        # 10) Color recovery (background-aware), and DO NOT recover where alpha is tiny
        # C = (B - (1-a)*BgB) / a
        alpha_safe = np.maximum(alpha, 1e-3)[..., None]
        out_rgb = (B - (1.0 - alpha)[..., None] * BgB) / alpha_safe
        out_rgb = np.clip(out_rgb, 0.0, 255.0)

        # zero RGB where alpha is effectively 0 to avoid colored dust
        out_rgb = np.where((alpha[..., None] <= noise_cut), 0.0, out_rgb)
        out_rgb_u8 = out_rgb.astype(np.uint8)

        out_a_u8 = alpha_to_u8(alpha)

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
    <h2>Difference Matting Test (Pro-stabilized)</h2>
    <p>Upload a <b>white</b> and a <b>black</b> background version of the same artwork (pixel-aligned).</p>

    <form action="/matte/difference?max_side=3000&bg_blur=24&noise_cut=0.02&fg_thresh=0.25&keep_near_fg_px=4&speck_alpha_max=0.25&min_speck_area=150&edge_soften=1&soften_sigma=0.9&hole_remove=0&min_hole_area=800"
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
      <li><b>noise_cut</b> (0.02–0.05): higher removes more tiny dust, but can eat very soft edges.</li>
      <li><b>min_speck_area</b> (80–300): higher removes bigger specks.</li>
      <li><b>bg_blur</b> (16–32): higher handles gradients better, too high can slightly affect edge math.</li>
      <li><b>soften_sigma</b> (0.8–1.2): only affects the edge band (won’t smear background).</li>
      <li><b>hole_remove</b>: enable only if you see unwanted transparent holes inside the design.</li>
    </ul>

    <p style="color:#777;">
      If results look too “cut out”, lower <b>noise_cut</b> and/or <b>min_speck_area</b>.
      If you still see confetti, raise <b>noise_cut</b> slightly (e.g. 0.03) and/or increase <b>min_speck_area</b>.
    </p>
  </body>
</html>
"""

@app.get("/health")
async def health():
    return {"status": "ok", "endpoint": "/matte/difference"}
