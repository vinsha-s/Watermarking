"""Command-line execution of the full HSI watermarking pipeline."""
import argparse
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from config import (DATASETS, WATERMARK_PATH, OUTPUT_DIR, K_S, N, K, NEIGH,
                    GAUSSIAN_SIGMA, W_EDGE, W_CONTRAST, W_UNIQUE, W_STABILITY,
                    WAVELET, DWT_MODE, DWT_LEVEL, DELTA, TAU_H)
from io_utils import load_largest_3d, normalize_bandwise, load_rgb_watermark
from watermark_encryption import encrypt_watermark, decrypt_watermark
from band_selection import compute_band_scores
from band_pairing import correlation_pairing
from watermark_embedding import embed_payload
from watermark_extraction import extract_payload
from evaluation import (imperceptibility_rows, nc_rgb, ber_rgb, ATTACKS,
                        attack_selected_bands, auth_decision)

def run_dataset(dataset_name: str, mat_path: str, W_rgb: np.ndarray):
    print("\n" + "="*80)
    print("DATASET:", dataset_name)
    print("="*80)

    key, cube = load_largest_3d(mat_path)
    cube_norm, _, _ = normalize_bandwise(cube)
    print("MAT variable:", key, "| shape:", cube_norm.shape)

    # Useful reviewer check for the Pavia identity issue.
    if dataset_name == "Pavia Centre":
        print("NOTE: verify that this exact file/shape is Pavia Centre, not Pavia University.")

    # 1) Encrypt watermark
    We, hW, enc_diag = encrypt_watermark(W_rgb, K_S, TAU_H)
    print("Encrypted payload bits:", We.size)
    print("h_W (SHA-256):", hW.hex())
    print("Hologram clipping diagnostics:", enc_diag)

    # 2) Band selection + pairing
    bs = compute_band_scores(cube_norm)
    corr, pairs = correlation_pairing(cube_norm, bs["selected_zero"])
    print("Selected bands (1-based):", bs["selected_one"].tolist())
    print("Pairs (1-based):", [(p+1, q+1, round(c, 6)) for p,q,c in pairs])

    # 3) Embed four chunks
    wm_bands, pair_meta = embed_payload(cube_norm, pairs, We, DELTA)

    # 4) Imperceptibility on exactly the eight modified/selected bands
    imp = imperceptibility_rows(dataset_name, cube_norm, wm_bands)
    imp_df = pd.DataFrame(imp)
    imp_df.to_csv(os.path.join(OUTPUT_DIR, f"{dataset_name}_selected_band_imperceptibility.csv".replace(" ","_")), index=False)
    print("Selected-band average PSNR:", imp_df["PSNR"].mean())
    print("Selected-band average SSIM:", imp_df["SSIM"].mean())

    # 5) No-attack full extraction/decryption baseline
    We0 = extract_payload(wm_bands, pairs, We.size, DELTA)
    W0 = decrypt_watermark(We0, K_S, hW, TAU_H)
    nc0 = nc_rgb(W_rgb, W0)
    ber0 = ber_rgb(W_rgb, W0)
    payload_ber0 = float(np.mean(We != We0))
    print(f"NO ATTACK: RGB NC={nc0:.6f}, RGB BER={ber0:.6f}, encrypted-payload BER={payload_ber0:.6f}")

    # 6) Corrected 20-attack evaluation
    rows = []
    for ai, (attack_name, fn) in enumerate(ATTACKS):
        received = attack_selected_bands(wm_bands, fn, ai)
        We_hat = extract_payload(received, pairs, We.size, DELTA)
        W_hat = decrypt_watermark(We_hat, K_S, hW, TAU_H)

        nc = nc_rgb(W_rgb, W_hat)             # Eq. 59, RGB values
        ber = ber_rgb(W_rgb, W_hat)           # Eq. 60, full 8-bit RGB bits
        payload_ber = float(np.mean(We != We_hat))

        rows.append({
            "Dataset": dataset_name,
            "Attack": attack_name,
            "NC_RGB_Eq59": nc,
            "BER_payload": payload_ber,
            "BER_RGB_diagnostic": ber,
            "Auth_decision": auth_decision(nc, payload_ber),
        })
        print(f"{attack_name:28s} NC={nc:.6f}  BER={ber:.6f}  payloadBER={payload_ber:.6f}")

    attack_df = pd.DataFrame(rows)
    attack_df.to_csv(os.path.join(OUTPUT_DIR, f"{dataset_name}_corrected_20_attacks.csv".replace(" ","_")), index=False)

    # Side information / reproducibility record (secret K_S intentionally excluded)
    side = {
        "dataset": dataset_name,
        "mat_variable": key,
        "cube_shape": list(cube_norm.shape),
        "watermark_shape": list(W_rgb.shape),
        "h_W_hex": hW.hex(),
        "K": K,
        "selected_bands_zero_based": [int(x) for x in bs["selected_zero"]],
        "selected_bands_one_based": [int(x) for x in bs["selected_one"]],
        "pairs_zero_based": [[int(p),int(q)] for p,q,_ in pairs],
        "pairs_one_based": [[int(p+1),int(q+1)] for p,q,_ in pairs],
        "pair_correlations": [float(c) for _,_,c in pairs],
        "weights": [W_EDGE, W_CONTRAST, W_UNIQUE, W_STABILITY],
        "NEIGH": NEIGH,
        "GAUSSIAN_SIGMA": GAUSSIAN_SIGMA,
        "wavelet": WAVELET,
        "dwt_mode": DWT_MODE,
        "dwt_level": DWT_LEVEL,
        "DELTA": DELTA,
        "TAU_H": TAU_H,
        "payload_bits": int(We.size),
        "chunk_bits": int(We.size // len(pairs)),
        "coefficient_traversal": "row-major d1 then row-major d2",
        "PRNG_and_keystream": "SHA-256 counter-mode deterministic expansion",
        "pair_metadata": pair_meta,
        "no_attack_NC_RGB": nc0,
        "no_attack_BER_RGB_diagnostic": ber0,
        "no_attack_BER_payload": payload_ber0,
        "hologram_diagnostics": enc_diag,
    }
    with open(os.path.join(OUTPUT_DIR, f"{dataset_name}_side_information.json".replace(" ","_")), "w") as f:
        json.dump(side, f, indent=2)

    # Save no-attack recovered watermark for visual verification.
    Image.fromarray(W0, mode="RGB").save(os.path.join(OUTPUT_DIR, f"{dataset_name}_recovered_no_attack.png".replace(" ","_")))

    return {
        "dataset": dataset_name,
        "attack_df": attack_df,
        "imperceptibility_df": imp_df,
        "no_attack": {"NC": nc0, "BER_RGB_diagnostic": ber0, "payload_BER": payload_ber0},
        "side": side,
    }


def main():
    W_rgb = load_rgb_watermark(WATERMARK_PATH)
    print("Original RGB watermark:", W_rgb.shape, W_rgb.dtype)
    print("DELTA =", DELTA, "| TAU_H =", TAU_H, "| wavelet =", WAVELET, "| mode =", DWT_MODE)

    results = []
    for ds, path in DATASETS.items():
        results.append(run_dataset(ds, path, W_rgb))

    # Computed attack summary (NC for recovered RGB, BER for encrypted payload).
    # Not claimed to reproduce the published tables without original inputs.
    merged = None
    for r in results:
        ds = r["dataset"]
        x = r["attack_df"][["Attack", "NC_RGB_Eq59", "BER_payload"]].copy()
        x = x.rename(columns={
            "NC_RGB_Eq59": f"{ds}_NC",
            "BER_payload": f"{ds}_payload_BER",
        })
        merged = x if merged is None else merged.merge(x, on="Attack", how="outer")
    merged.to_csv(os.path.join(OUTPUT_DIR, "ATTACKS_NC_PAYLOAD_BER.csv"), index=False)

    # Descriptive attack averages (not the manuscript's published table).
    avg_rows = []
    for r in results:
        df = r["attack_df"]
        avg_rows.append({
            "Dataset": r["dataset"],
            "Average_NC": float(df["NC_RGB_Eq59"].mean()),
            "Average_payload_BER": float(df["BER_payload"].mean()),
        })
    avg_df = pd.DataFrame(avg_rows)
    avg_df.to_csv(os.path.join(OUTPUT_DIR, "ATTACK_AVERAGES.csv"), index=False)

    # No-attack table requested by reviewer.
    no_attack_df = pd.DataFrame([
        {
            "Dataset": r["dataset"],
            "NC_no_attack": r["no_attack"]["NC"],
            "BER_payload_no_attack": r["no_attack"]["payload_BER"],
            "BER_RGB_diagnostic_no_attack": r["no_attack"]["BER_RGB_diagnostic"],
        }
        for r in results
    ])
    no_attack_df.to_csv(os.path.join(OUTPUT_DIR, "NO_ATTACK_RECOVERY.csv"), index=False)

    print("\n" + "="*80)
    print("COMPUTED 20-ATTACK AVERAGES")
    print(avg_df.to_string(index=False))
    print("\nCORRECTED TABLE 3")
    print(merged.to_string(index=False))
    print("\nNO-ATTACK RECOVERY")
    print(no_attack_df.to_string(index=False))
    print("\nSaved all outputs to:", OUTPUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holography-encrypted cross-band HSI watermarking")
    parser.add_argument("--watermark", default=WATERMARK_PATH, help="Rights-cleared RGB watermark image")
    parser.add_argument("--pavia", default=DATASETS["Pavia Centre"])
    parser.add_argument("--indian-pines", default=DATASETS["Indian Pines"])
    parser.add_argument("--salinas", default=DATASETS["Salinas"])
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--secret-env", default="HSI_WATERMARK_SECRET", help="Environment variable containing experimental secret")
    args = parser.parse_args()
    WATERMARK_PATH = args.watermark
    OUTPUT_DIR = args.output
    DATASETS = {"Pavia Centre": args.pavia, "Indian Pines": args.indian_pines, "Salinas": args.salinas}
    secret = os.getenv(args.secret_env)
    if not secret:
        parser.error(f"Set environment variable {args.secret_env} to a nonempty experimental secret")
    K_S = secret.encode("utf-8")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
