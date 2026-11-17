"""HSI watermarking source: band selection (refactored without altering computation)."""

import numpy as np
from scipy.ndimage import sobel, gaussian_filter
from config import EPS, NEIGH, GAUSSIAN_SIGMA, W_EDGE, W_CONTRAST, W_UNIQUE, W_STABILITY, K

def minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return (v - np.min(v)) / (np.max(v) - np.min(v) + EPS)


def compute_band_scores(cube_norm: np.ndarray):
    H, W, B = cube_norm.shape
    edge = np.zeros(B, dtype=np.float64)
    contrast = np.zeros(B, dtype=np.float64)
    unique = np.zeros(B, dtype=np.float64)
    hf_penalty = np.zeros(B, dtype=np.float64)

    for b in range(B):
        I = cube_norm[:, :, b].astype(np.float64)

        gx = sobel(I, axis=1, mode="reflect")
        gy = sobel(I, axis=0, mode="reflect")
        edge[b] = np.mean(np.hypot(gx, gy))
        contrast[b] = np.var(I)

        diffs = []
        for d in range(1, NEIGH + 1):
            if b-d >= 0:
                diffs.append(np.mean(np.abs(I - cube_norm[:, :, b-d])))
            if b+d < B:
                diffs.append(np.mean(np.abs(I - cube_norm[:, :, b+d])))
        unique[b] = np.mean(diffs) if diffs else 0.0

        smooth = gaussian_filter(I, sigma=GAUSSIAN_SIGMA, mode="reflect")
        residual = I - smooth
        hf_penalty[b] = np.median(np.abs(residual))

    edge_n = minmax(edge)
    contrast_n = minmax(contrast)
    unique_n = minmax(unique)
    stability_n = 1.0 - minmax(hf_penalty)

    final = (
        W_EDGE * edge_n +
        W_CONTRAST * contrast_n +
        W_UNIQUE * unique_n +
        W_STABILITY * stability_n
    )

    selected = np.argsort(final)[::-1][:K].astype(int)

    return {
        "edge": edge_n,
        "contrast": contrast_n,
        "uniqueness": unique_n,
        "stability": stability_n,
        "final": final,
        "selected_zero": selected,
        "selected_one": selected + 1,
    }
