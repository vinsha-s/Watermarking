# Holography-encrypted cross-band watermarking for hyperspectral images

Python source code organised by algorithm stage for the manuscript *Holography Encrypted Watermarking for Hyperspectral Images Using a Cross-Band Embedding Technique*.

## Source files

- `main.py`: End-to-end driver, data loading and computed outputs
- `config.py`: Parameters and example relative input paths
- `io_utils.py`: RGB watermark and HSI input, bandwise normalization
- `watermark_encryption.py`: Holographic RGB watermark encryption and decryption
- `band_selection.py`: Four-criterion Top-K spectral band selection
- `band_pairing.py`: Correlation-based greedy pairing
- `watermark_embedding.py`: DWT, cross-band differential maps, modulo-4 QIM embedding
- `watermark_extraction.py`: Blind recovery of the encrypted payload
- `evaluation.py`: Quality metrics, attacks, and authentication assessment
- `requirements.txt`: Python package dependencies

## Setup

```bash
python -m pip install -r requirements.txt
```

Provide the three `.mat` hyperspectral data files locally and an RGB watermark image; datasets and images are not part of this code distribution. The default paths in `config.py` are `data/Pavia.mat`, `data/Indian_pines.mat`, `data/Salinas_corrected.mat`, and `data/watermark.png`.

Set an experimental secret via the `HSI_WATERMARK_SECRET` environment variable. Use the same value on both sender and receiver sides. Example:

```bash
export HSI_WATERMARK_SECRET='your-experimental-secret'
python main.py --watermark data/watermark.png --output outputs
```

On Windows PowerShell:

```powershell
$env:HSI_WATERMARK_SECRET='your-experimental-secret'
python main.py --watermark data/watermark.png --output outputs
```

Optional arguments: `--pavia`, `--indian-pines`, `--salinas`, `--secret-env`.

## Notes

This is a modular refactoring of the previously provided single-file Python implementation. The computational functions and published parameter choices in that source have not been intentionally changed. It includes a 20-attack evaluation routine, but not all experimental comparisons or ablation studies reported in the manuscript. Full numerical reproduction requires the original inputs and experimental settings, and has not been independently verified. In particular, do not claim matching manuscript results without running and checking them.
