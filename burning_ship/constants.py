"""
Shared constants for the Burning Ship fractal viewer.

All configuration values, thresholds, encoding parameters, color
constants, and size presets live here so that other modules can
import them without pulling in pygame or heavy dependencies.
"""

from burning_ship_engine import (
    DiscoveryParams, Rect,
    PROFILE_BASIC, PROFILE_ADVANCED, PROFILE_GREAT_WALL,
)

# ---------------------------------------------------------------------------
# Palette / escape-count rendering
# ---------------------------------------------------------------------------

PALETTE_SIZE = 256                # number of colors per palette (u8 range)
DEFAULT_MAX_ITER = 64             # default max iterations for rendering
MAX_ITER_MIN = 1                  # lower bound for user-adjustable max_iter
MAX_ITER_MAX = 100000             # upper bound for user-adjustable max_iter

# ---------------------------------------------------------------------------
# Default fractal viewport
# ---------------------------------------------------------------------------

DEFAULT_CENTER_RE = -0.5
DEFAULT_CENTER_IM = -0.5
VIEWPORT_BASE_SPAN = 4.0

# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

PANEL_HEIGHT = 220                # bottom panel height in pixels

# Status bar colors (semantic)
CLR_NEUTRAL   = (200, 200, 200)   # default / informational
CLR_SUCCESS   = (100, 255, 100)   # operation completed successfully
CLR_ERROR     = (255, 100, 100)   # error or rejected input
CLR_WARNING   = (255, 200, 100)   # warning or attention needed
CLR_PENDING   = (200, 200, 100)   # in-progress / waiting
CLR_INFO      = (200, 200, 255)   # informational highlight (save/load)
CLR_BIT_OK    = (200, 255, 200)   # manual bit accepted
CLR_ADVANCE   = (100, 255, 200)   # auto-advance (point committed)
CLR_STAGE_RDY = (180, 220, 140)   # stage 2 ready
CLR_STAGE_ACT = (140, 255, 140)   # stage 2 active

# Cursor blink period
CURSOR_BLINK_MS = 530

# Point selection proximity threshold (pixels)
POINT_CLICK_THRESHOLD_PX = 20

# Debug hex field width (8 hex chars = 32 bits)
DEBUG_HEX_FIELD_CHARS = 8

# Default BIP39 mnemonic pre-filled in the input field (for demo/testing).
# NOT a real wallet — do not use with actual funds.
DEFAULT_BIP39_MNEMONIC = (
    "never use this example because private key secret need phrase true random"
)

# ---------------------------------------------------------------------------
# Perturbation encoding parameters
# ---------------------------------------------------------------------------

# o, p, q are all 64-bit reservoirs of entropy that select a private
# stage-2 fractal.  They share a uniform encoding (32-bit Re magnitude +
# 32-bit Im magnitude with sign bits at positions 31 and 63).  The only
# operational difference is that p carries a baseline (P_BASELINE_EXP)
# which steers the additive shift away from the canonical formula's
# degenerate-tail region.  Listed below in alphabetic order.

# Orbit seed o — entropy reservoir; no baseline
O_MAGNITUDE_BITS = 31
O_SIGN_BIT_RE = 31
O_SIGN_BIT_IM = 63
O_MAGNITUDE_MIN_EXP = 3

# Additive perturbation p — entropy reservoir; baseline = 2^{-P_BASELINE_EXP}
# steers p away from the canonical-formula tail (degenerate leaf areas).
P_MAGNITUDE_BITS = 31
P_SIGN_BIT_RE = 31
P_SIGN_BIT_IM = 63
P_MAGNITUDE_MIN_EXP = 4
P_BASELINE_EXP = 3                # baseline = 2^{-3} = 1/8

# Linear perturbation q (εz term) — entropy reservoir; no baseline
Q_MAGNITUDE_BITS = 31
Q_SIGN_BIT_RE = 31
Q_SIGN_BIT_IM = 63
Q_MAGNITUDE_MIN_EXP = 5

# Stage-1 fixed parameters: all three set to 0 yields the canonical
# Burning Ship formula (z₀=0, no additive shift, no linear term).
STAGE1_O = 0   # o=0 ⇒ orbit seed z₀ = 0
STAGE1_P = 0   # p=0 ⇒ additive shift only the baseline (+1/8, +1/8)
STAGE1_Q = 0   # q=0 ⇒ no εz term

# ---------------------------------------------------------------------------
# Argon2
# ---------------------------------------------------------------------------

ARGON2_INPUT_BYTES = 8

# ---------------------------------------------------------------------------
# Rendering parameters
# ---------------------------------------------------------------------------

# Leaf highlight boost
LEAF_BRIGHTNESS_BOOST = 1.4
LEAF_BRIGHTNESS_FLOOR = 0.1
LEAF_SATURATION_BOOST = 1.5
LEAF_SATURATION_THRESHOLD = 0.01

# Brightness falloff (sigmoid-like dimming)
BRIGHTNESS_FALLOFF_BASE = 16
BRIGHTNESS_EXPONENT_OFFSET = 4
BRIGHTNESS_OFFSET_STEP = 1.5

# Progressive rendering initial block size
PROGRESSIVE_INITIAL_BLOCK = 8

# Contraction formula: f(r) = (1 + 3r) / 4
CONTRACTION_MULTIPLIER = 3
CONTRACTION_DIVISOR = 4

# ---------------------------------------------------------------------------
# Size presets & encoding geometry
# ---------------------------------------------------------------------------

BITS_PER_POINT = 32

SIZE_PRESETS = {
    "mini":    {"points_per_stage": 1, "entropy_bits":  64, "bip39_words":  6},
    "default": {"points_per_stage": 2, "entropy_bits": 128, "bip39_words": 12},
    "large":   {"points_per_stage": 4, "entropy_bits": 256, "bip39_words": 24},
}
SIZE_PRESET_ORDER = ["mini", "default", "large"]
INITIAL_SIZE_PRESET = "default"

# Derived from initial preset (kept as module-level for backward compat)
STAGE1_NUM_POINTS = SIZE_PRESETS[INITIAL_SIZE_PRESET]["points_per_stage"]
STAGE2_NUM_POINTS = SIZE_PRESETS[INITIAL_SIZE_PRESET]["points_per_stage"]

# Encoding area: the BS region where island density supports 32-bit encoding
ENCODE_AREA = Rect.from_f64(-2.5, 1.5, -2.0, 1.5)

# ---------------------------------------------------------------------------
# Discovery params for GUI (faster than defaults)
# ---------------------------------------------------------------------------

GUI_PARAMS = DiscoveryParams(
    max_iter=DEFAULT_MAX_ITER,
    target_good=32,
    max_flood_points=256,
    min_grid_cells=1024*1024,
    p_max_shift=3,
    exclusion_threshold_num=1023,
    rng_seed=0x42,
)
