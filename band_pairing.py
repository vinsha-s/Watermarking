"""HSI watermarking source: band pairing (refactored without altering computation)."""

import numpy as np

def correlation_pairing(cube_norm: np.ndarray, selected_zero: np.ndarray):
    selected = [int(x) for x in selected_zero]
    if len(selected) % 2:
        raise ValueError("K must be even.")

    vectors = np.asarray([cube_norm[:, :, b].reshape(-1) for b in selected])
    corr = np.corrcoef(vectors)
    remaining = list(range(len(selected)))
    pairs = []

    while len(remaining) >= 2:
        best = None
        for a in range(len(remaining)):
            for b in range(a+1, len(remaining)):
                i, j = remaining[a], remaining[b]
                val = float(corr[i, j])
                if best is None or val > best[0]:
                    best = (val, i, j)
        val, i, j = best
        pairs.append((selected[i], selected[j], val))
        remaining.remove(i)
        remaining.remove(j)

    return corr, pairs
