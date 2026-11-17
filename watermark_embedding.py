"""HSI watermarking source: watermark embedding (refactored without altering computation)."""

from typing import Dict
import numpy as np
import pywt
from config import WAVELET, DWT_MODE, DWT_LEVEL, K

def dwt2_level2(I: np.ndarray):
    coeffs = pywt.wavedec2(I, wavelet=WAVELET, mode=DWT_MODE, level=DWT_LEVEL)
    LL2 = coeffs[0]
    LH2, HL2, HH2 = coeffs[1]
    LH1, HL1, HH1 = coeffs[2]
    return coeffs, LL2, LH2, HL2, HH2, LH1, HL1, HH1


def idwt2_from_parts(LL2, LH2, HL2, HH2, LH1, HL1, HH1, shape_hw):
    coeffs = [LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]
    out = pywt.waverec2(coeffs, wavelet=WAVELET, mode=DWT_MODE)
    return np.clip(out[:shape_hw[0], :shape_hw[1]], 0.0, 1.0).astype(np.float32)


def bits_to_symbols2(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size % 2 != 0:
        raise ValueError("2-bit symbol conversion requires an even bit count.")
    b = bits.reshape(-1, 2)
    return (2*b[:, 0] + b[:, 1]).astype(np.int64)


def symbols2_to_bits(symbols: np.ndarray) -> np.ndarray:
    s = np.asarray(symbols, dtype=np.int64).reshape(-1)
    return np.column_stack(((s >> 1) & 1, s & 1)).reshape(-1).astype(np.uint8)


def qim_target(d: np.ndarray, symbols: np.ndarray, delta: float) -> np.ndarray:
    """Eq. 28-29: nearest q*delta where q mod 4 == symbol."""
    x = np.asarray(d, dtype=np.float64)
    s = np.asarray(symbols, dtype=np.int64)
    k = np.rint((x / delta - s) / 4.0).astype(np.int64)
    q = s + 4 * k
    return q.astype(np.float64) * delta


def embed_pair_crossband(Ip: np.ndarray, Iq: np.ndarray, chunk_bits: np.ndarray, delta: float):
    cp, LLp, LHp, HLp, HHp, LH1p, HL1p, HH1p = dwt2_level2(Ip)
    cq, LLq, LHq, HLq, HHq, LH1q, HL1q, HH1q = dwt2_level2(Iq)

    if LHp.shape != HLq.shape or HLp.shape != LHq.shape:
        raise ValueError("Paired level-2 detail subband shapes do not match.")

    symbols = bits_to_symbols2(chunk_bits)
    cap_map = LHp.size
    cap_symbols = 2 * cap_map
    if symbols.size > cap_symbols:
        raise ValueError(
            f"Pair capacity insufficient: need {symbols.size*2} bits, "
            f"capacity={cap_symbols*2} bits, detail={LHp.shape}"
        )

    # Paper's fixed coefficient traversal order made explicit:
    # row-major d1 first, followed by row-major d2.
    n1 = min(symbols.size, cap_map)
    n2 = symbols.size - n1

    LHp_w = LHp.copy()
    HLp_w = HLp.copy()
    LHq_w = LHq.copy()
    HLq_w = HLq.copy()

    # d1 = LH_p - HL_q
    if n1 > 0:
        d1 = (LHp - HLq).reshape(-1)
        old = d1[:n1]
        target = qim_target(old, symbols[:n1], delta)
        dd = target - old
        a = LHp_w.reshape(-1)
        b = HLq_w.reshape(-1)
        a[:n1] += 0.5 * dd
        b[:n1] -= 0.5 * dd

    # d2 = HL_p - LH_q
    if n2 > 0:
        d2 = (HLp - LHq).reshape(-1)
        old = d2[:n2]
        target = qim_target(old, symbols[n1:n1+n2], delta)
        dd = target - old
        a = HLp_w.reshape(-1)
        b = LHq_w.reshape(-1)
        a[:n2] += 0.5 * dd
        b[:n2] -= 0.5 * dd

    Ip_w = idwt2_from_parts(LLp, LHp_w, HLp_w, HHp, LH1p, HL1p, HH1p, Ip.shape)
    Iq_w = idwt2_from_parts(LLq, LHq_w, HLq_w, HHq, LH1q, HL1q, HH1q, Iq.shape)

    return Ip_w, Iq_w, {
        "detail_shape": list(LHp.shape),
        "capacity_bits": int(4 * LHp.size),
        "used_bits": int(len(chunk_bits)),
        "d1_symbols": int(n1),
        "d2_symbols": int(n2),
        "traversal": "row-major d1 then row-major d2",
    }


def embed_payload(cube_norm: np.ndarray, pairs, We: np.ndarray, delta: float):
    M = len(pairs)
    if M != K // 2:
        raise ValueError(f"Expected {K//2} pairs for K={K}, got {M}")
    if We.size % M != 0:
        raise ValueError("Payload length must divide equally across band pairs.")

    chunks = np.split(We, M)
    wm_bands: Dict[int, np.ndarray] = {}
    pair_meta = []

    for m, ((bp, bq, corr), chunk) in enumerate(zip(pairs, chunks), start=1):
        Ip = cube_norm[:, :, bp]
        Iq = cube_norm[:, :, bq]
        Ip_w, Iq_w, meta = embed_pair_crossband(Ip, Iq, chunk, delta)
        wm_bands[bp] = Ip_w
        wm_bands[bq] = Iq_w
        meta.update({
            "pair_index": m,
            "band_p_zero": int(bp),
            "band_q_zero": int(bq),
            "band_p_one": int(bp+1),
            "band_q_one": int(bq+1),
            "correlation": float(corr),
            "chunk_bits": int(chunk.size),
        })
        pair_meta.append(meta)

    return wm_bands, pair_meta
