"""Default parameters and input paths. Exact data and secret set at runtime."""

EPS = 1e-12

# ================================================================
# 0. USER PATHS / PARAMETERS
# ================================================================

DATASETS = {
    "Pavia Centre": "data/Pavia.mat",
    "Indian Pines": "data/Indian_pines.mat",
    "Salinas": "data/Salinas_corrected.mat",
}

# Input images are not redistributed. Provide datasets and a rights-cleared RGB
# watermark locally. An input larger than 32x32 is resized to 32x32.
WATERMARK_PATH = "data/watermark.png"
OUTPUT_DIR = "outputs"
# An experimental secret must be supplied explicitly; do not publish real keys.
K_S = None

# Paper: N=32, K=8, equal four-criterion weights.
N = 32
K = 8
NEIGH = 2
GAUSSIAN_SIGMA = 1.0
W_EDGE = 0.25
W_CONTRAST = 0.25
W_UNIQUE = 0.25
W_STABILITY = 0.25

# DWT settings specified for this reference implementation.
WAVELET = "db2"
DWT_MODE = "periodization"
DWT_LEVEL = 2

# QIM step specified in the manuscript.
DELTA = 0.05

# Hologram clipping bound stated in the manuscript.
TAU_H = 0.75

# Reproducible attack noise seed.
ATTACK_SEED = 20261026

# Reported manuscript calibration thresholds. Calibration itself is not
# reproduced in this release; use only for comparing with published decisions.
TAU_NC = 0.730119
TAU_BER = 0.267524

# ================================================================
# 1. BASIC I/O / NORMALIZATION
# ================================================================

