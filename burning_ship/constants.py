"""
Shared constants for the Burning Ship fractal viewer.

All configuration values, thresholds, encoding parameters, color
constants, and size presets live here so that other modules can
import them without pulling in pygame or heavy dependencies.
"""

from burning_ship_engine import (
    DiscoveryParams, Rect,
    PROFILE_BASIC, PROFILE_ADVANCED, PROFILE_GREAT_WALL,
    ARGON2ID_MASTER_OUTPUT_BYTES,
)

# ---------------------------------------------------------------------------
# Palette / escape-count rendering
# ---------------------------------------------------------------------------

PALETTE_SIZE = 256                # number of colors per palette (u8 range)
DEFAULT_MAX_ITER = 64             # default max iterations for *rendering* (visual)
MAX_ITER_MIN = 1                  # lower bound for user-adjustable max_iter
MAX_ITER_MAX = 100000             # upper bound for user-adjustable max_iter

# Escape-iteration cap used for ENCODE/DECODE island discovery (GUI_PARAMS).
# This is intentionally decoupled from (and much larger than) the render cap:
# as the bisection tree zooms toward the Burning Ship set boundary, escape
# counts climb toward the cap, and a low cap (e.g. 64) makes almost every sample
# read as "non-escaping", starving island discovery — deep levels then burn
# 100k+ samples and the encode effectively stalls.  1024 (the engine's own
# DiscoveryParams default) keeps boundary-adjacent points escaping so discovery
# stays fast at every level.  escape_count short-circuits on escape, so the
# higher cap costs extra work only for the rare genuinely-non-escaping samples.
ENCODE_MAX_ITER = 1024

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

# Base (all-zero) parameters: o=p=q=0 yields the unperturbed Burning Ship
# formula (z₀=0, no additive shift beyond p's baseline, no linear term).
#
# NOTE (protocol 0.3.0): this is *no longer a privileged "canonical" first
# fractal*.  Every point stage — including the first — derives its (o, p, q)
# from the memory-hard chain seeded by stage-0 text (see protocol.py), so there
# is no public surface an attacker knows in advance.  These constants are kept
# only for the DEPRECATED two-stage helpers in encoding.py; the live pipeline
# never uses them.
STAGE1_O = 0   # o=0 ⇒ orbit seed z₀ = 0
STAGE1_P = 0   # p=0 ⇒ additive shift only the baseline (+1/8, +1/8)
STAGE1_Q = 0   # q=0 ⇒ no εz term
CANONICAL_O = STAGE1_O   # back-compat alias (deprecated; see note above)
CANONICAL_P = STAGE1_P
CANONICAL_Q = STAGE1_Q

# ---------------------------------------------------------------------------
# Argon2
# ---------------------------------------------------------------------------

ARGON2_INPUT_BYTES = 8

# ---------------------------------------------------------------------------
# Master-secret export (protocol 0.3.0)
# ---------------------------------------------------------------------------
#
# The master-secret export is a single Argon2id pass over the reproducible setup
# transcript (DESIGN.md "Master-Secret Export"): Argon2id, m = 2^16 KiB (64 MiB),
# p = 2, t = 8, fixed salt b"greatwall", output l = 1024 bytes.  The Argon2id
# parameters live in the Rust engine (argon2_hash.rs::argon2id_master);
# ARGON2ID_MASTER_OUTPUT_BYTES (the output length l) is re-exported here.
#
# Output-size ergonomics (TODO, deferred): 1024 bytes is unwieldy as a default.
# The intended UX gates the full output behind advanced options and shows a
# conventional 32 characters by default.  For the time being the export surfaces
# only the first MASTER_DISPLAY_CHARS hex characters of the Argon2id output.
MASTER_DISPLAY_CHARS = 32

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

# Protocol geometry (0.3.0): a mandatory, point-less STAGE 0 carries only a
# short text input, then exactly ONE 32-bit point (one needle) is encoded per
# later stage, and each point stage is its own fractal (one haystack).  The
# number of POINT stages is N = entropy_bits / BITS_PER_POINT; the total stage
# count is N + 1 because stage 0 is always present.  Every point stage's fractal
# is derived by hashing stage-0 text plus all preceding points through the
# memory-hard chain (Argon2 → SHA-256 → (o,p,q)).  There is NO canonical
# fractal: even the first point stage derives from stage-0 text, so all N
# haystacks are secret and chain-derived.
#
# Below, `n_stages` is the count of POINT stages (N) — it does NOT include
# stage 0.  Because N = entropy_bits / 32 = words / 3, every BIP39 size that is
# a multiple of 32 entropy bits falls out uniformly — one extra point stage per
# extra 32 bits (3 words).  We expose all of them, from 32 bits (3 words, N=1)
# up to a HARD CAP of 256 bits (24 words, N=8).  Going beyond 256 is allowed in
# theory but deliberately NOT offered: one more stage is the same marginal
# mental effort for diminishing returns, so the better lever past 24 words is
# more between-stage Argon2 iterations, not more stages.  (Future: a user with a
# specific reason could chain multiple setups via an "advanced pepper" field —
# pepper = a prior setup's result.  See great-wall-docs next-steps; revisit.)
#
# Tiers: 32/64/96 bits are "sub-standard" (below BIP39's 128-bit floor — weak,
# offered for completeness); 128..256 are the "standard" BIP39 sizes.
MAX_ENTROPY_BITS = 256                 # hard cap (24 words, N=8 point stages)
MIN_ENTROPY_BITS = BITS_PER_POINT      # 32 bits (3 words, N=1 point stage)


def _build_size_presets():
    """Build the full set of presets (one per 32-bit step up to the cap).

    Keyed by word count, e.g. "12w".  word_count = 3 * n_stages, and
    entropy_bits = 32 * n_stages.
    """
    presets = {}
    order = []
    for bits in range(MIN_ENTROPY_BITS, MAX_ENTROPY_BITS + 1, BITS_PER_POINT):
        n_stages = bits // BITS_PER_POINT
        words = 3 * n_stages
        key = f"{words}w"
        presets[key] = {
            "n_stages": n_stages,
            "entropy_bits": bits,
            "bip39_words": words,
            "tier": "standard" if bits >= 128 else "sub-standard",
        }
        order.append(key)
    return presets, order


SIZE_PRESETS, SIZE_PRESET_ORDER = _build_size_presets()
INITIAL_SIZE_PRESET = "12w"            # 128-bit / 12-word default

# Back-compat aliases for the three originally-named presets.
SIZE_PRESET_ALIASES = {"mini": "6w", "default": "12w", "large": "24w"}


def n_stages_for(entropy_bits):
    """Number of POINT stages (= one point each) for a given entropy-bit count.

    This is N (stages 1..N in the documented numbering); it does not count the
    mandatory point-less stage 0.
    """
    return entropy_bits // BITS_PER_POINT

# Encoding area: the BS region where island density supports 32-bit encoding
ENCODE_AREA = Rect.from_f64(-2.5, 1.5, -2.0, 1.5)

# ---------------------------------------------------------------------------
# Discovery params for GUI (faster than defaults)
# ---------------------------------------------------------------------------

GUI_PARAMS = DiscoveryParams(
    max_iter=ENCODE_MAX_ITER,
    target_good=32,
    max_flood_points=256,
    min_grid_cells=1024*1024,
    p_max_shift=3,
    exclusion_threshold_num=1023,
    rng_seed=0x42,
)
