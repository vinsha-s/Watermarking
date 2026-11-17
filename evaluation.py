"""HSI watermarking source: evaluation (refactored without altering computation)."""

from typing import Dict
import io
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,median_filter
from skimage.metrics import structural_similarity as ssim
from config import EPS, N, ATTACK_SEED, TAU_NC, TAU_BER

def nc_rgb(W: np.ndarray, W_hat: np.ndarray) -> float:
    A = np.asarray(W, dtype=np.float64)
    B = np.asarray(W_hat, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError(f"NC shape mismatch: {A.shape} vs {B.shape}")
    num = np.sum(A * B)
    den = np.sqrt(np.sum(A*A) * np.sum(B*B)) + EPS
    return float(num / den)


def ber_rgb(W: np.ndarray, W_hat: np.ndarray) -> float:
    A = np.asarray(W, dtype=np.uint8)
    B = np.asarray(W_hat, dtype=np.uint8)
    if A.shape != B.shape:
        raise ValueError(f"BER shape mismatch: {A.shape} vs {B.shape}")
    ba = np.unpackbits(A.reshape(-1))
    bb = np.unpackbits(B.reshape(-1))
    assert ba.size == N*N*3*8 == 24576
    return float(np.mean(ba != bb))


def psnr01(a, b):
    mse = float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return float("inf") if mse <= 0 else float(10*np.log10(1.0/mse))


def mae01(a, b):
    return float(np.mean(np.abs(a.astype(np.float64)-b.astype(np.float64))))


def mse01(a, b):
    return float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))


def entropy256(img01):
    u8 = np.clip(np.rint(img01*255.0), 0, 255).astype(np.uint8)
    hist = np.bincount(u8.reshape(-1), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + EPS)
    p = p[p > 0]
    return float(-np.sum(p*np.log2(p)))


def imperceptibility_rows(dataset_name, cube_norm, wm_bands):
    rows = []
    for b, Iw in sorted(wm_bands.items()):
        I = cube_norm[:, :, b]
        rows.append({
            "Dataset": dataset_name,
            "Band_zero": int(b),
            "Band_one": int(b+1),
            "PSNR": psnr01(I, Iw),
            "SSIM": float(ssim(I, Iw, data_range=1.0)),
            "MSE": mse01(I, Iw),
            "MAE": mae01(I, Iw),
            "Entropy_WM": entropy256(Iw),
        })
    return rows


def to_pil_gray(img01):
    return Image.fromarray(np.clip(np.rint(img01*255), 0, 255).astype(np.uint8), mode="L")


def from_pil_gray(im, shape):
    if im.size != (shape[1], shape[0]):
        im = im.resize((shape[1], shape[0]), Image.Resampling.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def attack_gaussian_noise(x, sigma, rng):
    return np.clip(x + rng.normal(0, sigma, x.shape), 0, 1).astype(np.float32)


def attack_salt_pepper(x, amount, rng):
    y = x.copy()
    n = int(round(amount * y.size))
    idx = rng.choice(y.size, size=min(n, y.size), replace=False)
    h = len(idx)//2
    f = y.reshape(-1)
    f[idx[:h]] = 0.0
    f[idx[h:]] = 1.0
    return y.astype(np.float32)


def attack_speckle(x, sigma, rng):
    return np.clip(x + x*rng.normal(0, sigma, x.shape), 0, 1).astype(np.float32)


def attack_jpeg(x, quality):
    buf = io.BytesIO()
    to_pil_gray(x).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return from_pil_gray(Image.open(buf).convert("L"), x.shape)


def attack_rotation(x, angle):
    # No hidden inverse alignment/synchronization is applied.
    im = to_pil_gray(x).rotate(float(angle), resample=Image.Resampling.BICUBIC,
                               expand=False, fillcolor=0)
    return from_pil_gray(im, x.shape)


def attack_scaling(x, scale):
    # Standard resize attack followed by resizing back to the original dimensions
    # so the HSI band array keeps its required shape. This is not an inverse
    # synchronization procedure; it is the definition of the scaling distortion.
    H, W = x.shape
    im = to_pil_gray(x)
    nw, nh = max(1, round(W*scale)), max(1, round(H*scale))
    im = im.resize((nw, nh), Image.Resampling.BICUBIC)
    im = im.resize((W, H), Image.Resampling.BICUBIC)
    return from_pil_gray(im, x.shape)


def attack_translation(x, tx, ty):
    im = to_pil_gray(x)
    # PIL inverse mapping: positive tx/ty here produces the manuscript's nominal shift.
    im2 = im.transform(im.size, Image.Transform.AFFINE,
                       (1, 0, -tx, 0, 1, -ty),
                       resample=Image.Resampling.BICUBIC, fillcolor=0)
    return from_pil_gray(im2, x.shape)


def attack_center_crop(x, ratio):
    H, W = x.shape
    ch, cw = max(1, round(H*ratio)), max(1, round(W*ratio))
    y0, x0 = (H-ch)//2, (W-cw)//2
    crop = x[y0:y0+ch, x0:x0+cw]
    im = to_pil_gray(crop).resize((W, H), Image.Resampling.BICUBIC)
    return from_pil_gray(im, x.shape)


def attack_sharpen(x):
    blur = gaussian_filter(x, sigma=1.0, mode="reflect")
    return np.clip(x + (x - blur), 0, 1).astype(np.float32)


ATTACKS = [
    ("Gaussian Noise (sigma=0.01)", lambda x,r: attack_gaussian_noise(x, 0.01, r)),
    ("Gaussian Noise (sigma=0.03)", lambda x,r: attack_gaussian_noise(x, 0.03, r)),
    ("Salt & Pepper (a=0.01)",      lambda x,r: attack_salt_pepper(x, 0.01, r)),
    ("Salt & Pepper (a=0.03)",      lambda x,r: attack_salt_pepper(x, 0.03, r)),
    ("Speckle (sigma=0.05)",        lambda x,r: attack_speckle(x, 0.05, r)),
    ("Speckle (sigma=0.10)",        lambda x,r: attack_speckle(x, 0.10, r)),
    ("Gaussian Blur (sigma=0.6)",   lambda x,r: gaussian_filter(x, 0.6, mode="reflect").astype(np.float32)),
    ("Gaussian Blur (sigma=1.2)",   lambda x,r: gaussian_filter(x, 1.2, mode="reflect").astype(np.float32)),
    ("Median (k=3)",                lambda x,r: median_filter(x, 3, mode="reflect").astype(np.float32)),
    ("Median (k=5)",                lambda x,r: median_filter(x, 5, mode="reflect").astype(np.float32)),
    ("JPEG (Q=90)",                 lambda x,r: attack_jpeg(x, 90)),
    ("JPEG (Q=70)",                 lambda x,r: attack_jpeg(x, 70)),
    ("JPEG (Q=50)",                 lambda x,r: attack_jpeg(x, 50)),
    ("Rotation (5 deg)",            lambda x,r: attack_rotation(x, 5)),
    ("Rotation (10 deg)",           lambda x,r: attack_rotation(x, 10)),
    ("Scaling (0.9x)",              lambda x,r: attack_scaling(x, 0.9)),
    ("Scaling (1.1x)",              lambda x,r: attack_scaling(x, 1.1)),
    ("Translation (5,5)",           lambda x,r: attack_translation(x, 5, 5)),
    ("Center Crop (0.9)",           lambda x,r: attack_center_crop(x, 0.9)),
    ("Sharpen (unsharp)",           lambda x,r: attack_sharpen(x)),
]


def attack_selected_bands(wm_bands: Dict[int, np.ndarray], attack_fn, attack_index: int):
    out = {}
    for j, (b, img) in enumerate(sorted(wm_bands.items())):
        rng = np.random.default_rng(ATTACK_SEED + 1000*attack_index + 17*j + b)
        out[b] = np.clip(attack_fn(img, rng), 0.0, 1.0).astype(np.float32)
    return out


def auth_decision(nc, ber):
    if TAU_NC is None or TAU_BER is None:
        return "Not evaluated (thresholds not calibrated)"
    return "Authentic" if (nc >= TAU_NC and ber <= TAU_BER) else "Not authentic"
