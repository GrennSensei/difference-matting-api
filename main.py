import io
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from PIL import Image, ImageFilter
import numpy as np
from collections import deque

app = FastAPI(title="Difference Matting API", version="1.0.0")


# ---------- Utilities ----------

def load_upload_as_rgb(file: UploadFile) -> Image.Image:
    data = file.file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img

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

def soften_alpha(alpha: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return alpha
    return alpha.filter(ImageFilter.GaussianBlur(radius=sigma))

def remove_small_holes(alpha_np: np.ndarray, min_area: int) -> np.ndarray:
    """
    Fill small enclosed transparent holes in alpha.
    alpha_np: uint8 [H,W]
    """
    if min_area <= 0:
        return alpha_np

    h, w = alpha_np.shape
    visited = np.zeros((h, w), dtype=np.uint8)

    def neighbors(y, x):
        if y > 0: yield y-1, x
        if y < h-1: yield y+1, x
        if x > 0: yield y, x-1
        if x < w-1: yield y, x+1

    # mark border-connected transparent pixels
    q = deque()
    for x in range(w):
        if alpha_np[0, x] == 0: q.append((0, x))
        if alpha_np[h-1, x] == 0: q.append((h-1, x))
    for y in range(h):
        if alpha_np[y, 0] == 0: q.append((y, 0))
        if alpha_np[y, w-1] == 0: q.append((y, w-1))

    while q:
        y, x = q.popleft()
        if visited[y, x]:
            continue
        if alpha_np[y, x] != 0:
            continue
        visited[y, x] = 1
        for ny, nx in neighbors(y, x):
            if not visited[ny, nx] and alpha_np[ny, nx] == 0:
                q.append((ny, nx))

    # find enclosed holes and fill if small
    hole_vis = np.zeros((h, w), dtype=np.uint8)
    for y0 in range(h):
        for x0 in range(w):
            if alpha_np[y0, x0] != 0:
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
                    if alpha_np[ny, nx] == 0 and visited[ny, nx] == 0 and hole_vis[ny, nx] == 0:
                        hole_vis[ny, nx] = 1
                        qq.append((ny, nx))

            if len(comp) <= min_area:
                for (yy, xx) in comp:
                    alpha_np[yy, xx] = 255

    return alpha_np


# ---------- Difference Matting Endpoint ----------

@app.post("/matte/difference")
async def difference_matte(
    white_file: UploadFile = File(..., description="Same artwork on pure white #FFFFFF background"),
    black_file: UploadFile = File(..., description="Same artwork on pure black #000000 background"),
    max_side: int = Query(3000, ge=512, le=6000),
    edge_soften: int = Query(1, ge=0, le=1),
    soften_sigma: float = Query(0.8, ge=0.0, le=10.0),
    hole_remove: int = Query(0, ge=0, le=1),
    min_hole_area: int = Query(800, ge=0, le=500000),
):
    img_w = resize_max(load_upload_as_rgb(white_file), max_side=max_side)
    img_b = resize_max(load_upload_as_rgb(black_file), max_side=max_side)

    if img_w.size != img_b.size:
        return JSONResponse(
            status_code=400,
            content={"error": "Dimension mismatch: White and black images must have identical size."}
        )

    # Convert to arrays
    W = np.array(img_w, dtype=np.uint8).astype(np.float32)  # [H,W,3]
    B = np.array(img_b, dtype=np.uint8).astype(np.float32)

    # alpha = 1 - ||W - B|| / ||white-black||
    diff = W - B
    pixel_dist = np.sqrt((diff * diff).sum(axis=2))             # [H,W]
    bg_dist = np.sqrt(3.0 * (255.0 ** 2))                        # ~441.67

    alpha = 1.0 - (pixel_dist / bg_dist)
    alpha = np.clip(alpha, 0.0, 1.0)

    # Recover color using black background assumption: B = C * alpha
    alpha_safe = np.maximum(alpha, 1e-3)[..., None]
    out_rgb = B / alpha_safe
    out_rgb = np.clip(out_rgb, 0.0, 255.0).astype(np.uint8)

    out_a = (alpha * 255.0).astype(np.uint8)

    if hole_remove == 1 and min_hole_area > 0:
        out_a = remove_small_holes(out_a, min_area=min_hole_area)

    alpha_img = Image.fromarray(out_a, mode="L")

    if edge_soften == 1 and soften_sigma > 0:
        alpha_img = soften_alpha(alpha_img, soften_sigma)

    out = Image.fromarray(out_rgb, mode="RGB").convert("RGBA")
    out.putalpha(alpha_img)

    return StreamingResponse(io.BytesIO(png_bytes(out)), media_type="image/png")


# ---------- Web test page ----------

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Difference Matting Test</title></head>
  <body style="font-family: Arial; max-width: 900px; margin: 40px auto; line-height:1.4;">
    <h2>Difference Matting Test</h2>
    <p>Upload a <b>white background</b> and a <b>black background</b> version of the same artwork.</p>
    <p><b>Critical:</b> The two images must be pixel-aligned (same size, same placement).</p>

    <form action="/matte/difference?max_side=3000&edge_soften=1&soften_sigma=0.8&hole_remove=0&min_hole_area=800"
          method="post" enctype="multipart/form-data"
          style="padding:16px; border:1px solid #ddd; border-radius:10px;">
      <p><b>White image (#FFFFFF background):</b></p>
      <input type="file" name="white_file" accept="image/*" required />

      <p style="margin-top:14px;"><b>Black image (#000000 background):</b></p>
      <input type="file" name="black_file" accept="image/*" required />

      <p style="margin-top:14px;">
        <button type="submit" style="padding:10px 14px; font-weight:bold; cursor:pointer;">
          Generate Transparent PNG
        </button>
      </p>
    </form>

    <p style="color:#666; margin-top:14px;">
      Defaults: edge_soften=1 (sigma 0.8). If you see noise, try increasing soften_sigma to 1.2.
      If inner holes remain, enable hole_remove=1 and lower min_hole_area (e.g. 300–800).
    </p>
  </body>
</html>
"""

@app.get("/health")
async def health():
    return {"status": "ok", "endpoint": "/matte/difference"}
