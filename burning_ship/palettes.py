"""
Color palettes and escape-count transform functions for the fractal viewer.
"""

import math
import colorsys

import numpy as np

from constants import PALETTE_SIZE

# ---------------------------------------------------------------------------
# Color schemes
# ---------------------------------------------------------------------------

def make_palette_classic(size=PALETTE_SIZE):
    """Deep blue/black classic Mandelbrot-style palette."""
    pal = [(0, 0, 0)] * size
    for i in range(1, size):
        t = i / size
        r = int(9 * (1 - t) * t * t * t * 255)
        g = int(15 * (1 - t) * (1 - t) * t * t * 255)
        b = int(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255)
        pal[i] = (min(r, 255), min(g, 255), min(b, 255))
    return pal


def make_palette_fire(size=PALETTE_SIZE):
    """Fire / heat map palette."""
    pal = [(0, 0, 0)] * size
    for i in range(1, size):
        t = i / size
        r = min(int(t * 3 * 255), 255)
        g = min(int((t - 0.33) * 3 * 255), 255) if t > 0.33 else 0
        b = min(int((t - 0.66) * 3 * 255), 255) if t > 0.66 else 0
        pal[i] = (r, g, b)
    return pal


def make_palette_ice(size=PALETTE_SIZE):
    """Ice / cool blue palette."""
    pal = [(0, 0, 0)] * size
    for i in range(1, size):
        t = i / size
        r = min(int((t - 0.5) * 2 * 255), 255) if t > 0.5 else 0
        g = min(int(t * 2 * 255), 255)
        b = min(int(255 * (0.5 + 0.5 * math.sin(t * math.pi))), 255)
        pal[i] = (r, g, b)
    return pal


def make_palette_rainbow(size=PALETTE_SIZE):
    """HSV rainbow palette."""
    pal = [(0, 0, 0)] * size
    for i in range(1, size):
        h = (i / size) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 1.0)
        pal[i] = (int(r * 255), int(g * 255), int(b * 255))
    return pal


def make_palette_highcontrast(size=PALETTE_SIZE):
    """High-contrast palette with sharp color bands for easy visual distinction."""
    pal = [(0, 0, 0)] * size
    bands = [
        (255, 255, 255),  # white
        (255, 0, 0),      # red
        (0, 255, 0),      # green
        (0, 0, 255),      # blue
        (255, 255, 0),    # yellow
        (0, 255, 255),    # cyan
        (255, 0, 255),    # magenta
        (255, 128, 0),    # orange
    ]
    for i in range(1, size):
        pal[i] = bands[i % len(bands)]
    return pal


PALETTES = {
    "Classic": make_palette_classic(),
    "Fire": make_palette_fire(),
    "Ice": make_palette_ice(),
    "Rainbow": make_palette_rainbow(),
    "HiCon": make_palette_highcontrast(),
}
PALETTE_NAMES = list(PALETTES.keys())

# Pre-build numpy LUT arrays (PALETTE_SIZE x 3, uint8) for fast palette application
PALETTE_LUTS = {}
for _name, _pal in PALETTES.items():
    _lut = np.zeros((PALETTE_SIZE, 3), dtype=np.uint8)
    for _i, (_r, _g, _b) in enumerate(_pal):
        _lut[_i] = (_r, _g, _b)
    PALETTE_LUTS[_name] = _lut

# ---------------------------------------------------------------------------
# Escape-count transform functions (applied before color mapping)
# ---------------------------------------------------------------------------

def _esc_identity(d):
    return d

def _esc_square(d):
    return d * d

def _esc_cube(d):
    return d * d * d

def _esc_exp(d):
    return np.exp(d)

def _esc_sqrt(d):
    return np.sqrt(d)

def _esc_cbrt(d):
    return np.cbrt(d)

def _esc_log(d):
    return np.log1p(d)

ESC_TRANSFORMS = [
    ("Identity",  _esc_identity),
    ("Square",    _esc_square),
    ("Cube",      _esc_cube),
    ("Exp",       _esc_exp),
    ("Sqrt",      _esc_sqrt),
    ("Cbrt",      _esc_cbrt),
    ("Log",       _esc_log),
]
ESC_TRANSFORM_NAMES = [name for name, _ in ESC_TRANSFORMS]
