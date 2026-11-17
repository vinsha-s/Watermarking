"""HSI watermarking source: watermark encryption (refactored without altering computation)."""

import hashlib
import math
from typing import Tuple
import numpy as np
from config import N

def sha256_bytes(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def hash_expand(key: bytes, nbytes: int, domain: bytes) -> bytes:
    """SHA-256 counter-mode deterministic expansion.

    This fixes the manuscript's previously unspecified concrete PRNG/keystream
    implementation. Report it explicitly if used for the revision.
    """
    out = bytearray()
    counter = 0
    while len(out) < nbytes:
        ctr = counter.to_bytes(8, byteorder="big", signed=False)
        out.extend(hashlib.sha256(key + domain + ctr).digest())
        counter += 1
    return bytes(out[:nbytes])


def uniform_matrix_from_seed(seed: bytes, shape: Tuple[int, int], domain: bytes) -> np.ndarray:
    n = int(np.prod(shape))
    raw = hash_expand(seed, n * 8, domain)
    vals = np.frombuffer(raw, dtype=">u8").astype(np.float64)
    # Map uint64 exactly into [0,1).
    vals /= float(2**64)
    return vals.reshape(shape)


def keystream_bits_from_key(key: bytes, nbits: int) -> np.ndarray:
    raw = hash_expand(key, math.ceil(nbits / 8), b"keystream")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
    return bits.astype(np.uint8)


def watermark_digest(W_rgb: np.ndarray) -> bytes:
    # Fixed channel-major order: R channel, then G, then B; each row-major.
    bs = np.transpose(W_rgb, (2, 0, 1)).copy(order="C").tobytes()
    return hashlib.sha256(bs).digest()


def channel_parameters(Ks: bytes, hW: bytes, channel: str):
    c = channel.encode("ascii")
    seed_c = sha256_bytes(Ks, hW, c, b"seed")
    key_c = sha256_bytes(Ks, hW, c, b"key")
    return seed_c, key_c


def phase_masks(seed_c: bytes, N_: int):
    R1 = uniform_matrix_from_seed(seed_c, (N_, N_), b"phase-mask-1")
    R2 = uniform_matrix_from_seed(seed_c, (N_, N_), b"phase-mask-2")
    phi1 = 2.0 * np.pi * R1
    phi2 = 2.0 * np.pi * R2
    P1 = np.exp(1j * phi1)
    P2 = np.exp(1j * phi2)
    return P1, P2


def q3_to_bits(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.uint8).reshape(-1)
    return np.column_stack(((q >> 2) & 1, (q >> 1) & 1, q & 1)).astype(np.uint8)


def bits3_to_q(bits3: np.ndarray) -> np.ndarray:
    b = np.asarray(bits3, dtype=np.uint8)
    return (4*b[:, 0] + 2*b[:, 1] + b[:, 2]).astype(np.uint8)


def encrypt_watermark(W_rgb: np.ndarray, Ks: bytes, tau_H: float):
    if W_rgb.shape != (N, N, 3):
        raise ValueError(f"Expected watermark {(N,N,3)}, got {W_rgb.shape}")
    if tau_H <= 0:
        raise ValueError("TAU_H must be positive.")

    hW = watermark_digest(W_rgb)
    enc_channels = []
    diagnostics = {}

    for ci, cname in enumerate(["R", "G", "B"]):
        Wc = W_rgb[:, :, ci].astype(np.float64)
        Ac = Wc / 255.0

        seed_c, key_c = channel_parameters(Ks, hW, cname)
        P1, P2 = phase_masks(seed_c, N)

        # Eqs. 4-7
        Uc = Ac * P1
        Fc = np.fft.fft2(Uc)
        Gc = Fc * P2
        Hc = np.fft.ifft2(Gc)

        Hr = np.real(Hc)
        Hi = np.imag(Hc)

        # Eqs. 9-11: clipping and 3-bit uniform quantization
        Hr_n = np.clip(Hr / tau_H, -1.0, 1.0)
        Hi_n = np.clip(Hi / tau_H, -1.0, 1.0)
        Qr = np.rint(7.0 * (Hr_n + 1.0) / 2.0).astype(np.uint8)
        Qi = np.rint(7.0 * (Hi_n + 1.0) / 2.0).astype(np.uint8)
        Qr = np.clip(Qr, 0, 7)
        Qi = np.clip(Qi, 0, 7)

        # Eq. 12-14: 3 real bits + 3 imag bits per coefficient, row-wise
        br = q3_to_bits(Qr)
        bi = q3_to_bits(Qi)
        Bc = np.concatenate([br, bi], axis=1).reshape(-1).astype(np.uint8)
        assert Bc.size == 6 * N * N

        # Eq. 15: XOR diffusion
        KS_c = keystream_bits_from_key(key_c, Bc.size)
        Bc_enc = np.bitwise_xor(Bc, KS_c).astype(np.uint8)
        enc_channels.append(Bc_enc)

        diagnostics[cname] = {
            "max_abs_real": float(np.max(np.abs(Hr))),
            "max_abs_imag": float(np.max(np.abs(Hi))),
            "clip_fraction_real": float(np.mean(np.abs(Hr) > tau_H)),
            "clip_fraction_imag": float(np.mean(np.abs(Hi) > tau_H)),
        }

    We = np.concatenate(enc_channels).astype(np.uint8)
    assert We.size == 3 * 6 * N * N == 18432
    return We, hW, diagnostics


def decrypt_watermark(We_hat: np.ndarray, Ks: bytes, hW: bytes, tau_H: float) -> np.ndarray:
    We_hat = np.asarray(We_hat, dtype=np.uint8).reshape(-1)
    expected = 3 * 6 * N * N
    if We_hat.size != expected:
        raise ValueError(f"Encrypted payload length must be {expected}, got {We_hat.size}")

    channel_len = 6 * N * N
    recovered = []

    for ci, cname in enumerate(["R", "G", "B"]):
        Benc = We_hat[ci*channel_len:(ci+1)*channel_len]
        seed_c, key_c = channel_parameters(Ks, hW, cname)
        P1, P2 = phase_masks(seed_c, N)

        KS_c = keystream_bits_from_key(key_c, channel_len)
        Bc = np.bitwise_xor(Benc, KS_c).astype(np.uint8)

        groups = Bc.reshape(N*N, 6)
        Qr = bits3_to_q(groups[:, :3]).reshape(N, N)
        Qi = bits3_to_q(groups[:, 3:]).reshape(N, N)

        # Eq. 51-52
        Hr = tau_H * (2.0 * Qr.astype(np.float64) / 7.0 - 1.0)
        Hi = tau_H * (2.0 * Qi.astype(np.float64) / 7.0 - 1.0)
        Hc = Hr + 1j * Hi

        # Eqs. 53-56
        Gc = np.fft.fft2(Hc)
        Fc = Gc * np.conj(P2)
        Uc = np.fft.ifft2(Fc)
        Ac = np.real(Uc * np.conj(P1))

        # Eq. 57
        Ac = np.clip(Ac, 0.0, 1.0)
        Wc_hat = np.clip(np.rint(Ac * 255.0), 0, 255).astype(np.uint8)
        recovered.append(Wc_hat)

    return np.stack(recovered, axis=2)
