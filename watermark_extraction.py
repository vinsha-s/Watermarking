"""HSI watermarking source: watermark extraction (refactored without altering computation)."""

from typing import Dict
import numpy as np
from watermark_embedding import dwt2_level2, symbols2_to_bits

def extract_pair_crossband(Ip_r: np.ndarray, Iq_r: np.ndarray, n_chunk_bits: int, delta: float):
    _, _, LHp, HLp, _, _, _, _ = dwt2_level2(Ip_r)
    _, _, LHq, HLq, _, _, _, _ = dwt2_level2(Iq_r)

    nsym = n_chunk_bits // 2
    cap_map = LHp.size
    n1 = min(nsym, cap_map)
    n2 = nsym - n1

    recovered_symbols = []

    if n1 > 0:
        d1 = (LHp - HLq).reshape(-1)[:n1]
        q1 = np.rint(d1 / delta).astype(np.int64)
        s1 = np.mod(q1, 4)
        recovered_symbols.append(s1)

    if n2 > 0:
        d2 = (HLp - LHq).reshape(-1)[:n2]
        q2 = np.rint(d2 / delta).astype(np.int64)
        s2 = np.mod(q2, 4)
        recovered_symbols.append(s2)

    symbols = np.concatenate(recovered_symbols) if recovered_symbols else np.empty(0, dtype=int)
    bits = symbols2_to_bits(symbols)[:n_chunk_bits]
    return bits.astype(np.uint8)


def extract_payload(received_bands: Dict[int, np.ndarray], pairs, total_payload_bits: int, delta: float):
    M = len(pairs)
    if total_payload_bits % M != 0:
        raise ValueError("Payload length must divide equally across pairs.")
    chunk_len = total_payload_bits // M
    chunks = []
    for bp, bq, _corr in pairs:
        if bp not in received_bands or bq not in received_bands:
            raise KeyError(f"Missing received selected bands {bp}, {bq}")
        chunks.append(extract_pair_crossband(received_bands[bp], received_bands[bq], chunk_len, delta))
    We_hat = np.concatenate(chunks).astype(np.uint8)
    return We_hat[:total_payload_bits]
