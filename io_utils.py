"""HSI watermarking source: io utils (refactored without altering computation)."""

import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
from config import EPS, N

def load_largest_3d(mat_path: str):
    mat = loadmat(mat_path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    best_key, best_size = None, -1
    for k in keys:
        arr = mat[k]
        if hasattr(arr, "ndim") and arr.ndim == 3:
            size = int(np.prod(arr.shape))
            if size > best_size:
                best_key, best_size = k, size
    if best_key is None:
        raise ValueError(f"No 3-D HSI cube found in {mat_path}. Variables={keys}")
    return best_key, np.asarray(mat[best_key], dtype=np.float32)


def normalize_bandwise(cube: np.ndarray):
    cube = np.asarray(cube, dtype=np.float32)
    band_min = cube.min(axis=(0, 1), keepdims=True)
    band_max = cube.max(axis=(0, 1), keepdims=True)
    norm = (cube - band_min) / (band_max - band_min + EPS)
    return np.clip(norm, 0.0, 1.0).astype(np.float32), band_min, band_max


def load_rgb_watermark(path: str, size=(N, N)) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Watermark not found: {path}")
    im = Image.open(path).convert("RGB")
    if im.size != (size[1], size[0]):
        im = im.resize((size[1], size[0]), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.uint8)
