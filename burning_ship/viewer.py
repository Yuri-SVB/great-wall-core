#!/usr/bin/env python3
"""
Burning Ship fractal viewer with BIP39 seed encoding/decoding.

Controls:
  Mouse wheel / +/-      Zoom in/out at cursor
  Arrow keys / drag      Pan
  R                      Reset view
  1-5                    Switch color scheme
  Tab                    Toggle BIP39 input focus
  Enter                  Encode seed (when input focused)
  C                      Clear encoded points
  D                      Toggle debug mode (show hex logs)
  W                      Cycle size preset (mini 6w/64b, default 12w/128b, large 24w/256b)
  S                      Toggle select-points mode (click points to decode, stage 1)
  T                      Toggle between Stage 1 (d=2) and Stage 2 (generalized d)
  V                      Toggle area visualization for encoded/selected points
  , / .                  Navigate to previous/next bisection step (when V active)
  L                      Toggle brightness falloff (sigmoid cave-like dimming)
  X                      Stretch correction: click P1 then P2 to compress along that axis
  Z                      Clear all stretch corrections
  F2                     Random encode (stage 1 + Argon2 + stage 2)
  F5                     Load session from JSON
  F6                     Save session to JSON
  Escape / Q             Quit (Escape cancels stretch mode if active)
"""

import sys
import os
import ctypes
import math
import hashlib
import signal

os.environ.setdefault("SDL_VIDEODRIVER", "x11")

import pygame
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from burning_ship_engine import (
    encode, decode, decode_full, get_precision, DiscoveryParams, Rect,
    DEFAULT_AREA, render_viewport, render_viewport_generic,
    cache_init, cache_destroy, cache_clear,
    fixed_to_f64, fixed_from_f64, FIXED_ONE,
    cache_init_stage2, cache_destroy_stage2, cache_clear_stage2,
    _lib,
    PROFILE_BASIC, PROFILE_ADVANCED, PROFILE_GREAT_WALL,
)
from bip39 import (
    mnemonic_to_bits, bits_to_mnemonic,
    )
from constants import (
    PALETTE_SIZE, DEFAULT_MAX_ITER, MAX_ITER_MIN, MAX_ITER_MAX,
    DEFAULT_CENTER_RE, DEFAULT_CENTER_IM, VIEWPORT_BASE_SPAN,
    PANEL_HEIGHT,
    CLR_NEUTRAL, CLR_SUCCESS, CLR_ERROR, CLR_WARNING, CLR_PENDING,
    CLR_INFO, CLR_BIT_OK, CLR_ADVANCE, CLR_STAGE_RDY, CLR_STAGE_ACT,
    CURSOR_BLINK_MS, POINT_CLICK_THRESHOLD_PX, DEBUG_HEX_FIELD_CHARS,
    DEFAULT_BIP39_MNEMONIC,
    STAGE1_O, STAGE1_P, STAGE1_Q,
    LEAF_BRIGHTNESS_BOOST, LEAF_BRIGHTNESS_FLOOR,
    LEAF_SATURATION_BOOST, LEAF_SATURATION_THRESHOLD,
    BRIGHTNESS_FALLOFF_BASE, BRIGHTNESS_EXPONENT_OFFSET,
    BRIGHTNESS_OFFSET_STEP,
    PROGRESSIVE_INITIAL_BLOCK,
    CONTRACTION_MULTIPLIER, CONTRACTION_DIVISOR,
    BITS_PER_POINT, SIZE_PRESETS, SIZE_PRESET_ORDER, INITIAL_SIZE_PRESET,
    STAGE1_NUM_POINTS, STAGE2_NUM_POINTS,
    ENCODE_AREA, GUI_PARAMS,
)

from palettes import (
    PALETTES, PALETTE_NAMES, PALETTE_LUTS,
    ESC_TRANSFORMS, ESC_TRANSFORM_NAMES,
)
from encoding import (
    argon2_path_marker, encode_bip39, encode_bip39_stage2,
    decode_points, encode_bits_stage,
    bits_to_bytes, bits_to_hex, compute_checksum_bits,
)
from argon2_pipeline import (
    run_argon2_iterative, derive_stage2_params,
    decode_p_display, decode_q_display, decode_o_display,
    run_random_encode,
)
from session import (
    copy_to_clipboard, paste_from_clipboard,
    save_session, load_session,
)
from text_input import handle_text_input
from manual_mode import (
    manual_update_viz, manual_encode_latest,
    manual_add_bit,
)

# ---------------------------------------------------------------------------
# Viewer state
# ---------------------------------------------------------------------------

class ViewerState:
    def __init__(self, width=800, height=600):
        self.win_w = width
        self.win_h = height
        self.panel_h = PANEL_HEIGHT
        self.vp_w = width
        self.vp_h = height - self.panel_h

        # Size preset
        self.size_preset = INITIAL_SIZE_PRESET

        # Fractal view (complex plane coordinates)
        self.center_re = DEFAULT_CENTER_RE
        self.center_im = DEFAULT_CENTER_IM
        self.zoom_exp = 0  # zoom = sqrt(2)^zoom_exp; restricted to 2^k * sqrt(2)^b

        self.palette_idx = 0
        self.max_iter = DEFAULT_MAX_ITER
        self.maxiter_text = str(DEFAULT_MAX_ITER)
        self.maxiter_focused = False
        self.maxiter_cursor = len(self.maxiter_text)
        self.needs_redraw = True     # fractal data changed (triggers Rust render)
        self.needs_repalette = True  # palette changed (just re-apply colors)

        # BIP39
        self.input_text = DEFAULT_BIP39_MNEMONIC
        self.input_cursor = len(DEFAULT_BIP39_MNEMONIC)
        self.input_sel = self.input_cursor  # selection anchor (== cursor when no selection)
        self.input_focused = False
        self.salt_text = ""
        self.salt_cursor = 0
        self.salt_focused = False
        self.status_msg = "Press S to select points, D for debug mode"
        self.status_color = CLR_NEUTRAL

        # Encoded points (list of (re, im, re_raw, im_raw) tuples)
        # Stage 1 points (P1, P2) — first 64 entropy bits
        self.stage1_encoded_points = []
        self.stage1_encoded_bits_chunks = []
        self.stage1_encoded_steps = []
        self.stage1_encoded_final_rects = []
        # Stage 2 points (P3, P4) — last 64 entropy bits
        self.stage2_encoded_points = []
        self.stage2_encoded_bits_chunks = []
        self.stage2_encoded_steps = []
        self.stage2_encoded_final_rects = []

        # Debug: selected encoded point index (None = none selected)
        self.selected_point_idx = None
        # Selected decoded point index (None = none selected)
        self.selected_decoded_idx = None

        # Select-points mode
        self.select_mode = False
        self.selected_points = []  # stage-1: list of (re_raw, im_raw) tuples
        self.selected_steps = []   # list of step-data lists, one per decoded point
        self.selected_final_rects = []  # list of Rect, one per decoded point
        self.stage2_selected_points = []  # stage-2: list of (re_raw, im_raw) tuples
        self.stage2_selected_steps = []
        self.stage2_selected_final_rects = []
        self.decoded_mnemonic = ""
        self.decoded_stage1_bits = None  # 64 bits from stage-1 decode (for Argon2)
        self.decoded_stage2_bits = None  # 64 bits from stage-2 decode

        # Path prefix accumulated during point selection
        self._select_path_prefix = "O"

        # Manual bit input mode (M key toggles; O=0, I=1)
        self.manual_bits_mode = False
        self.manual_bits = []            # bits for current point (max BITS_PER_POINT)
        self.manual_point_idx = 0        # 0 or 1 (which of the 2 points in this stage)
        self.manual_encode_result = None # EncodeResult after each bit
        self.manual_encode_history = []  # cached EncodeResult per bit count (for instant undo)
        self.manual_path_prefix = "O"   # path prefix for current point
        self.manual_committed_points = []  # list of committed (re, im, re_raw, im_raw)
        self.manual_committed_steps = []   # list of step-data lists
        self.manual_committed_rects = []   # list of final Rect
        self.manual_committed_bits = []    # list of bit-lists, one per committed point

        # Area visualization (V key toggles; < > navigate steps)
        self.show_areas = False
        self.area_focus_step = None  # None = show all, int = highlight single step

        # Debug mode (D key toggles; hex logs only shown when True)
        self.debug_mode = False

        # Debug: manual hex perturbation inputs (visible only in debug mode)
        self.debug_p_re_hex = "0" * DEBUG_HEX_FIELD_CHARS
        self.debug_p_im_hex = "0" * DEBUG_HEX_FIELD_CHARS
        self.debug_p_re_focused = False
        self.debug_p_im_focused = False
        self.debug_p_re_cursor = DEBUG_HEX_FIELD_CHARS
        self.debug_p_im_cursor = DEBUG_HEX_FIELD_CHARS

        # Debug: manual hex linear perturbation inputs (q parameter)
        self.debug_q_re_hex = "0" * DEBUG_HEX_FIELD_CHARS
        self.debug_q_im_hex = "0" * DEBUG_HEX_FIELD_CHARS
        self.debug_q_re_focused = False
        self.debug_q_im_focused = False
        self.debug_q_re_cursor = DEBUG_HEX_FIELD_CHARS
        self.debug_q_im_cursor = DEBUG_HEX_FIELD_CHARS

        # Debug: manual hex orbit seed inputs (o parameter)
        self.debug_o_re_hex = "0" * DEBUG_HEX_FIELD_CHARS
        self.debug_o_im_hex = "0" * DEBUG_HEX_FIELD_CHARS
        self.debug_o_re_focused = False
        self.debug_o_im_focused = False
        self.debug_o_re_cursor = DEBUG_HEX_FIELD_CHARS
        self.debug_o_im_cursor = DEBUG_HEX_FIELD_CHARS

        # Argon2 hashing
        self.argon2_iterations = "0"       # text field for iteration count
        self.argon2_iter_focused = False   # whether the iterations field has focus
        self.argon2_iter_cursor = 4        # cursor position in iterations field
        self.argon2_digest = ""            # hex digest result
        self.argon2_running = False        # True while hash is computing
        self.argon2_progress = 0           # completed iterations so far
        self.argon2_progress_total = 0     # total iterations requested
        self.argon2_stage1_bits = None     # cached 64-bit list from encode/decode
        self.argon2_profile = PROFILE_BASIC  # 0=Basic, 1=Advanced, 2=Great Wall
        self.stage1_path = "O"              # accumulated path after stage 1
        self.argon2_marker = ""             # e.g. "B0", "A100"

        # Stage-2 fractal parameters (derived from Argon2 digest)
        self.stage2_o = None     # uint64 orbit seed encoding
        self.stage2_o_re = None  # float display value of Re(z₀)
        self.stage2_o_im = None  # float display value of Im(z₀)
        self.stage2_p = None     # uint64 additive perturbation encoding
        self.stage2_p_re = None  # float display value of Re(p)
        self.stage2_p_im = None  # float display value of Im(p)
        self.stage2_q = None     # uint64 linear perturbation encoding
        self.stage2_q_re = None  # float display value of Re(ε)
        self.stage2_q_im = None  # float display value of Im(ε)

        # Active rendering stage (1 = canonical d=2, 2 = perturbed)
        self.stage = 1

        # Brightness falloff (cave-exploration effect)
        self.brightness_falloff = True
        self.brightness_exponent_offset = BRIGHTNESS_EXPONENT_OFFSET

        # Escape-count transform index (into ESC_TRANSFORMS)
        self.esc_transform_idx = 0

        # Highlighted leaf area (Rect or None) — for visual feedback in select mode
        self.highlighted_leaf_rect = None

        # Progressive rendering: block sizes from coarse to fine
        self.progressive_block = None  # None = idle, int = current block size to render

        # Rendering buffer
        self.pixel_buf = np.zeros(self.vp_w * self.vp_h, dtype=np.uint8)

        # Stretch correction (X key to define, Z key to clear)
        # Each entry: (p1_re, p1_im, theta, scale) in complex-plane coords
        self.stretch_corrections = []
        self.stretch_mode = False       # True while picking P1/P2
        self.stretch_p1 = None          # (re, im) of first click, or None
        self._stretched_surface = None  # cached warped surface

    @property
    def zoom(self):
        """Zoom = sqrt(2)^zoom_exp, always of the form 2^k * sqrt(2)^b."""
        return math.sqrt(2) ** self.zoom_exp

    @property
    def points_per_stage(self):
        return SIZE_PRESETS[self.size_preset]["points_per_stage"]

    @property
    def entropy_bits(self):
        return SIZE_PRESETS[self.size_preset]["entropy_bits"]

    @property
    def bip39_words(self):
        return SIZE_PRESETS[self.size_preset]["bip39_words"]

    @property
    def bits_per_stage(self):
        return self.points_per_stage * BITS_PER_POINT

    @property
    def encoded_points(self):
        """Return encoded points for the current stage."""
        return self.stage2_encoded_points if self.stage == 2 else self.stage1_encoded_points

    @encoded_points.setter
    def encoded_points(self, value):
        if self.stage == 2:
            self.stage2_encoded_points = value
        else:
            self.stage1_encoded_points = value

    @property
    def encoded_bits_chunks(self):
        return self.stage2_encoded_bits_chunks if self.stage == 2 else self.stage1_encoded_bits_chunks

    @encoded_bits_chunks.setter
    def encoded_bits_chunks(self, value):
        if self.stage == 2:
            self.stage2_encoded_bits_chunks = value
        else:
            self.stage1_encoded_bits_chunks = value

    @property
    def encoded_steps(self):
        return self.stage2_encoded_steps if self.stage == 2 else self.stage1_encoded_steps

    @encoded_steps.setter
    def encoded_steps(self, value):
        if self.stage == 2:
            self.stage2_encoded_steps = value
        else:
            self.stage1_encoded_steps = value

    @property
    def encoded_final_rects(self):
        return self.stage2_encoded_final_rects if self.stage == 2 else self.stage1_encoded_final_rects

    @encoded_final_rects.setter
    def encoded_final_rects(self, value):
        if self.stage == 2:
            self.stage2_encoded_final_rects = value
        else:
            self.stage1_encoded_final_rects = value

    def reset_view(self):
        self.center_re = DEFAULT_CENTER_RE
        self.center_im = DEFAULT_CENTER_IM
        self.zoom_exp = 0
        self.needs_redraw = True

    def screen_to_complex(self, sx, sy):
        """Convert screen pixel to complex coordinate (f64, for rendering)."""
        step = self.pixel_step()
        origin_re, origin_im = self.viewport_origin()
        return origin_re + sx * step, origin_im + sy * step

    def screen_to_complex_fixed(self, sx, sy):
        """Convert screen pixel to raw Fixed i64 coordinates.

        All arithmetic is in fixed-point so that no f64 rounding can push the
        coordinate across a bisection split boundary.  The result matches the
        grid that the Rust renderer samples (to within 1 ULP from the integer
        step division).
        """
        # step_fixed = VIEWPORT_BASE_SPAN * FIXED_ONE / (vp_w * zoom)
        # zoom = sqrt(2)^zoom_exp.  Represent as (mantissa, shift) where
        #   zoom_exp even: zoom = 2^(zoom_exp//2)
        #   zoom_exp odd:  zoom = 2^(zoom_exp//2) * sqrt(2)
        # We compute step_fixed = base_span_fixed // (vp_w * zoom_fixed)
        # using 128-bit intermediates to avoid overflow.
        base_span_fixed = int(VIEWPORT_BASE_SPAN * FIXED_ONE)  # exact (4.0)
        zexp = self.zoom_exp
        half = zexp // 2
        # Numerator for step: base_span_fixed (this is pixel step * vp_w * zoom)
        # step_fixed = base_span_fixed / (vp_w * zoom)
        # With zoom = 2^half [* sqrt(2)]:
        if zexp % 2 == 0:
            # zoom = 2^half, so vp_w * zoom = vp_w << half  (or >> -half)
            if half >= 0:
                denom = self.vp_w << half
            else:
                denom = max(1, self.vp_w >> (-half))
            step_fixed = base_span_fixed // denom
        else:
            # zoom = 2^half * sqrt(2).
            # step = base_span / (vp_w * 2^half * sqrt(2))
            #      = base_span / (vp_w * 2^half) / sqrt(2)
            #      = (base_span / (vp_w * 2^half)) * sqrt(2) / 2
            # Compute in two stages to stay integer.
            if half >= 0:
                denom1 = self.vp_w << half
            else:
                denom1 = max(1, self.vp_w >> (-half))
            coarse = base_span_fixed // denom1
            # Multiply by sqrt(2)/2 ≈ 0.7071... via integer ratio
            # 3037000499/4294967296 ≈ sqrt(2)/2 to 32-bit precision
            step_fixed = (coarse * 3037000499) >> 32

        center_re_fixed = fixed_from_f64(self.center_re)
        center_im_fixed = fixed_from_f64(self.center_im)
        origin_re_fixed = center_re_fixed - (self.vp_w // 2) * step_fixed
        origin_im_fixed = center_im_fixed - (self.vp_h // 2) * step_fixed
        re_raw = origin_re_fixed + sx * step_fixed
        im_raw = origin_im_fixed + sy * step_fixed
        return re_raw, im_raw

    def complex_to_screen(self, re, im):
        """Convert complex coordinate to screen pixel."""
        step = self.pixel_step()
        origin_re, origin_im = self.viewport_origin()
        sx = (re - origin_re) / step
        sy = (im - origin_im) / step
        return int(sx), int(sy)

    def pixel_step(self):
        """Size of one pixel in complex-plane units."""
        base_span = VIEWPORT_BASE_SPAN
        return base_span / (self.vp_w * self.zoom)

    def viewport_origin(self):
        step = self.pixel_step()
        origin_re = self.center_re - (self.vp_w / 2) * step
        origin_im = self.center_im - (self.vp_h / 2) * step
        return origin_re, origin_im


def render_fractal(state, block_size=1):
    """Render fractal into pixel buffer.

    block_size > 1 renders at reduced resolution (w/block, h/block) then
    upscales by filling each block with the sampled value.  Used for
    progressive rendering: coarse pass first for responsiveness.
    """
    step = state.pixel_step()
    origin_re, origin_im = state.viewport_origin()

    # Always use the perturbed formula: stage 1 uses STAGE1_P (baseline),
    # stage 2 uses the Argon2-derived stage2_p.
    o_val = state.stage2_o if (state.stage == 2 and state.stage2_o is not None) else STAGE1_O
    p_val = state.stage2_p if (state.stage == 2 and state.stage2_p is not None) else STAGE1_P
    q_val = state.stage2_q if (state.stage == 2 and state.stage2_q is not None) else STAGE1_Q

    if block_size <= 1:
        # Full resolution render
        out_ptr = state.pixel_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        _lib.bs_render_viewport_generic(
            origin_re, origin_im, step,
            state.vp_w, state.vp_h, state.max_iter,
            o_val, p_val, q_val, out_ptr,
        )
    else:
        # Reduced resolution render, then upscale
        small_w = max(1, state.vp_w // block_size)
        small_h = max(1, state.vp_h // block_size)
        coarse_step = step * block_size
        small_buf = np.zeros(small_w * small_h, dtype=np.uint8)
        small_ptr = small_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        _lib.bs_render_viewport_generic(
            origin_re, origin_im, coarse_step,
            small_w, small_h, state.max_iter,
            o_val, p_val, q_val, small_ptr,
        )

        # Upscale: repeat each pixel into a block_size x block_size block
        small_2d = small_buf.reshape(small_h, small_w)
        upscaled = np.repeat(np.repeat(small_2d, block_size, axis=1), block_size, axis=0)
        # Crop/pad to exact viewport size
        out_2d = state.pixel_buf.reshape(state.vp_h, state.vp_w)
        uh, uw = upscaled.shape
        copy_h = min(uh, state.vp_h)
        copy_w = min(uw, state.vp_w)
        out_2d[:copy_h, :copy_w] = upscaled[:copy_h, :copy_w]


def apply_palette(pixel_buf, palette_name, w, h, state=None):
    """Convert escape-count buffer to RGB surface using numpy LUT.

    When state.brightness_falloff is True, each pixel's colour is scaled by
    a sigmoid-like dimming curve based on escape count and zoom.

    When a non-identity escape-count transform is active, the raw escape
    counts are passed through it and re-normalised to 1-255 before the
    palette LUT lookup.  Index 0 (non-escaping) is always preserved.
    """
    lut = PALETTE_LUTS[palette_name]
    raw = pixel_buf.reshape(h, w)

    # Apply escape-count transform (rendering only, floating point).
    transform_idx = state.esc_transform_idx if state is not None else 0
    if transform_idx != 0:
        _, tfn = ESC_TRANSFORMS[transform_idx]
        d = raw.astype(np.float64)
        mask = d > 0  # preserve 0 = non-escaping
        transformed = np.zeros_like(d)
        transformed[mask] = tfn(d[mask])
        # Normalise to 1-255 range
        t_min = transformed[mask].min() if mask.any() else 0
        t_max = transformed[mask].max() if mask.any() else 1
        if t_max > t_min:
            transformed[mask] = 1 + (transformed[mask] - t_min) / (t_max - t_min) * 254
        elif mask.any():
            transformed[mask] = 128
        indices = transformed.astype(np.uint8)
    else:
        indices = raw

    rgb = lut[indices].astype(np.float32)

    if state is not None and state.brightness_falloff:
        d = raw.astype(np.float32)
        z = state.zoom
        beo = state.brightness_exponent_offset
        factor = BRIGHTNESS_FALLOFF_BASE / (BRIGHTNESS_FALLOFF_BASE + (2**(d - beo))/(z*z))
        rgb *= factor[:, :, np.newaxis]

    # Highlight pixels inside the selected leaf area (higher saturation + brightness)
    if state is not None and state.highlighted_leaf_rect is not None:
        leaf = state.highlighted_leaf_rect
        step = state.pixel_step()
        origin_re, origin_im = state.viewport_origin()

        # Convert leaf rect to pixel coordinates
        col_min = int((leaf.re_min_f64() - origin_re) / step)
        col_max = int((leaf.re_max_f64() - origin_re) / step)
        row_min = int((leaf.im_min_f64() - origin_im) / step)
        row_max = int((leaf.im_max_f64() - origin_im) / step)

        # Clamp to viewport
        col_min = max(0, min(col_min, w))
        col_max = max(0, min(col_max, w))
        row_min = max(0, min(row_min, h))
        row_max = max(0, min(row_max, h))

        if col_max > col_min and row_max > row_min:
            patch = rgb[row_min:row_max, col_min:col_max]
            patch_norm = patch / 255.0
            cmax = patch_norm.max(axis=2)
            cmin = patch_norm.min(axis=2)
            delta = cmax - cmin
            sat = np.where(cmax > 0, delta / (cmax + 1e-10), 0.0)

            # Boost brightness with a floor lift
            new_val = np.clip(cmax * LEAF_BRIGHTNESS_BOOST + LEAF_BRIGHTNESS_FLOOR, 0, 1)
            brightness_scale = new_val / (cmax + 1e-10)
            boosted = patch_norm * brightness_scale[:, :, np.newaxis]

            # Boost saturation: pull channels away from their mean
            mean_ch = boosted.mean(axis=2, keepdims=True)
            sat_boost = np.where(sat[:, :, np.newaxis] > LEAF_SATURATION_THRESHOLD, LEAF_SATURATION_BOOST, 1.0)
            boosted = mean_ch + (boosted - mean_ch) * sat_boost

            patch[:] = np.clip(boosted * 255.0, 0, 255)

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    surface = pygame.Surface((w, h))
    arr = pygame.surfarray.pixels3d(surface)
    # rgb is (h, w, 3), surfarray wants (w, h, 3)
    np.copyto(arr, rgb.transpose(1, 0, 2))
    del arr
    return surface


# Fraction of viewport width used as the reference length C for stretch
# correction.  When |P1P2| == C the compression ratio is 1 (no change).
STRETCH_C_FRACTION = 0.25

# Maximum expansion factor for the stretched render buffer (per axis).
# Prevents excessive memory/time when compression is extreme.
STRETCH_MAX_EXPANSION = 4


def _inverse_warp_coords(w, h, state):
    """Compute source complex coordinates for every destination pixel.

    Returns (re, im) arrays of shape (h, w) — the complex-plane positions
    each destination pixel should sample from after undoing all stretch
    corrections.
    """
    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float64),
        np.arange(h, dtype=np.float64),
    )  # shape (h, w)
    step = state.pixel_step()
    origin_re, origin_im = state.viewport_origin()
    # Convert pixel grid to complex coordinates
    re = origin_re + gx * step
    im = origin_im + gy * step

    # Apply inverse of each correction (in reverse order to undo compounding)
    for p1_re, p1_im, theta, scale in reversed(state.stretch_corrections):
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        cos_nt = math.cos(-theta)
        sin_nt = math.sin(-theta)
        inv_scale = 1.0 / scale  # expand to invert the compression

        # Translate to P1 origin
        dx = re - p1_re
        dy = im - p1_im
        # Rotate by -theta (align stretch axis with x)
        rx = dx * cos_nt - dy * sin_nt
        ry = dx * sin_nt + dy * cos_nt
        # Expand x by inverse scale
        rx *= inv_scale
        # Rotate back by +theta
        dx2 = rx * cos_t - ry * sin_t
        dy2 = rx * sin_t + ry * cos_t
        # Translate back
        re = dx2 + p1_re
        im = dy2 + p1_im

    return re, im


def apply_stretch_corrections(surface, state):
    """Warp *surface* with stretch corrections, rendering extra area as needed.

    When the inverse warp maps destination pixels to source coordinates
    outside the current viewport, an expanded fractal region is rendered
    so that no edge-clamping artefacts appear.

    Returns a new pygame Surface (same size as input).
    """
    if not state.stretch_corrections:
        return surface

    w, h = state.vp_w, state.vp_h
    step = state.pixel_step()
    origin_re, origin_im = state.viewport_origin()

    # Compute where every destination pixel samples from
    src_re, src_im = _inverse_warp_coords(w, h, state)

    # Determine bounding box of needed source coordinates
    src_re_min = float(src_re.min())
    src_re_max = float(src_re.max())
    src_im_min = float(src_im.min())
    src_im_max = float(src_im.max())

    vp_re_max = origin_re + w * step
    vp_im_max = origin_im + h * step

    needs_expanded = (
        src_re_min < origin_re - step or src_re_max > vp_re_max + step or
        src_im_min < origin_im - step or src_im_max > vp_im_max + step
    )

    if needs_expanded:
        # Render an expanded fractal area that covers all source coords
        exp_w = int(math.ceil((src_re_max - src_re_min) / step)) + 2
        exp_h = int(math.ceil((src_im_max - src_im_min) / step)) + 2
        max_dim_w = w * STRETCH_MAX_EXPANSION
        max_dim_h = h * STRETCH_MAX_EXPANSION
        exp_w = min(exp_w, max_dim_w)
        exp_h = min(exp_h, max_dim_h)

        exp_buf = np.zeros(exp_w * exp_h, dtype=np.uint8)
        exp_ptr = exp_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        o_val = state.stage2_o if (state.stage == 2 and state.stage2_o is not None) else STAGE1_O
        p_val = state.stage2_p if (state.stage == 2 and state.stage2_p is not None) else STAGE1_P
        q_val = state.stage2_q if (state.stage == 2 and state.stage2_q is not None) else STAGE1_Q

        _lib.bs_render_viewport_generic(
            src_re_min, src_im_min, step,
            exp_w, exp_h, state.max_iter,
            o_val, p_val, q_val, exp_ptr,
        )

        # Apply palette to expanded buffer (skip leaf highlight)
        saved_leaf = state.highlighted_leaf_rect
        state.highlighted_leaf_rect = None
        exp_surface = apply_palette(
            exp_buf, PALETTE_NAMES[state.palette_idx],
            exp_w, exp_h, state,
        )
        state.highlighted_leaf_rect = saved_leaf

        # Map source complex coords → expanded-buffer pixel coords
        src_px_x = ((src_re - src_re_min) / step).clip(0, exp_w - 1).astype(np.int32)
        src_px_y = ((src_im - src_im_min) / step).clip(0, exp_h - 1).astype(np.int32)

        src_arr = pygame.surfarray.pixels3d(exp_surface)
    else:
        # All source coords fit within the already-rendered viewport
        src_px_x = ((src_re - origin_re) / step).clip(0, w - 1).astype(np.int32)
        src_px_y = ((src_im - origin_im) / step).clip(0, h - 1).astype(np.int32)

        src_arr = pygame.surfarray.pixels3d(surface)

    # Resample: surfarray is (W, H, 3); src arrays are (h, w) → transpose
    out_surface = pygame.Surface((w, h))
    out_arr = pygame.surfarray.pixels3d(out_surface)
    out_arr[:, :, :] = src_arr[src_px_x.T, src_px_y.T, :]
    del src_arr
    del out_arr
    return out_surface


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_marker(screen, sx, sy, color, label="", font=None):
    """Draw a crosshair marker at screen position."""
    size = 8
    pygame.draw.line(screen, color, (sx - size, sy), (sx + size, sy), 2)
    pygame.draw.line(screen, color, (sx, sy - size), (sx, sy + size), 2)
    pygame.draw.circle(screen, color, (sx, sy), size, 2)
    if label and font:
        txt = font.render(label, True, color)
        screen.blit(txt, (sx + size + 2, sy - 8))


MARKER_COLORS = [
    (255, 80, 80),    # red
    (80, 255, 80),    # green
    (80, 180, 255),   # blue
    (255, 255, 80),   # yellow
]

# Colors for bisection depth levels (cycle through these)
DEPTH_COLORS = [
    (255, 255, 255),
    (255, 200, 80),
    (80, 255, 200),
    (200, 80, 255),
    (255, 80, 160),
    (80, 200, 255),
    (200, 255, 80),
    (255, 160, 80),
]


def _all_selected_steps(state):
    """Return the combined selected steps list (stage 1 + stage 2)."""
    return state.selected_steps + state.stage2_selected_steps


def _all_selected_final_rects(state):
    """Return the combined selected final rects list (stage 1 + stage 2)."""
    return state.selected_final_rects + state.stage2_selected_final_rects


def _area_total_steps(state):
    """Total number of bisection steps across the active point set."""
    if state.selected_point_idx is not None and state.selected_point_idx < len(state.encoded_steps):
        return len(state.encoded_steps[state.selected_point_idx])
    all_sel = _all_selected_steps(state)
    if state.selected_decoded_idx is not None and state.selected_decoded_idx < len(all_sel):
        return len(all_sel[state.selected_decoded_idx])
    # Use encoded steps if available, otherwise selected (decoded) steps
    steps_list = state.encoded_steps if state.encoded_steps else all_sel
    return sum(len(s) for s in steps_list)


def _get_active_steps(state):
    """Return list of (steps, final_rect, color) tuples for area visualization."""
    result = []
    # If a single encoded point is focused, show only that
    if state.selected_point_idx is not None and state.selected_point_idx < len(state.encoded_steps):
        idx = state.selected_point_idx
        result.append((
            state.encoded_steps[idx],
            state.encoded_final_rects[idx],
            MARKER_COLORS[idx % len(MARKER_COLORS)],
        ))
        return result
    # If a single decoded point is focused, show only that (across both stages)
    all_sel = _all_selected_steps(state)
    all_sel_rects = _all_selected_final_rects(state)
    if state.selected_decoded_idx is not None and state.selected_decoded_idx < len(all_sel):
        idx = state.selected_decoded_idx
        result.append((
            all_sel[idx],
            all_sel_rects[idx],
            MARKER_COLORS[idx % len(MARKER_COLORS)],
        ))
        return result
    # Encoded points first
    for i, (steps, frect) in enumerate(zip(state.encoded_steps, state.encoded_final_rects)):
        result.append((steps, frect, MARKER_COLORS[i % len(MARKER_COLORS)]))
    # Then selected (decoded) points — all stages
    for i, (steps, frect) in enumerate(zip(all_sel, all_sel_rects)):
        result.append((steps, frect, MARKER_COLORS[i % len(MARKER_COLORS)]))
    return result


def draw_bisect_rects(screen, state, steps, final_rect, point_color,
                      focus_step=None):
    """Draw the bisection rectangle hierarchy for a selected point.

    Draws each step's area rectangle and the split line, from root to leaf.
    If focus_step is an int, only that step is drawn (highlighted).
    """
    vp_clip = pygame.Rect(0, 0, state.vp_w, state.vp_h)

    for i, step in enumerate(steps):
        if focus_step is not None and i != focus_step:
            continue
        area = step['area']
        color = DEPTH_COLORS[i % len(DEPTH_COLORS)]
        is_focused = (focus_step is not None)

        # Convert rect corners to screen coords (raw Fixed → f64 for display)
        x1, y1 = state.complex_to_screen(area.re_min_f64(), area.im_min_f64())
        x2, y2 = state.complex_to_screen(area.re_max_f64(), area.im_max_f64())
        rx = min(x1, x2)
        ry = min(y1, y2)
        rw = abs(x2 - x1)
        rh = abs(y2 - y1)

        # Skip if completely off screen or too small
        rect = pygame.Rect(rx, ry, max(rw, 1), max(rh, 1))
        if not rect.colliderect(vp_clip):
            continue

        # Draw rectangle outline (thicker when focused)
        thickness = 3 if is_focused else (2 if rw > 4 and rh > 4 else 1)
        pygame.draw.rect(screen, color, rect, thickness)

        # Draw split line
        if step['split_vertical']:
            sx, _ = state.complex_to_screen(step['split_coord_f64'], 0)
            if vp_clip.left <= sx <= vp_clip.right:
                lw = 2 if is_focused else 1
                pygame.draw.line(screen, color, (sx, max(ry, 0)), (sx, min(ry + rh, state.vp_h)), lw)
        else:
            _, sy = state.complex_to_screen(0, step['split_coord_f64'])
            if vp_clip.top <= sy <= vp_clip.bottom:
                lw = 2 if is_focused else 1
                pygame.draw.line(screen, color, (max(rx, 0), sy), (min(rx + rw, state.vp_w), sy), lw)

        # Draw island barycenters as small diamonds
        for isl in step.get('islands', []):
            ix, iy = state.complex_to_screen(isl['center_re'], isl['center_im'])
            if 0 <= ix < state.vp_w and 0 <= iy < state.vp_h:
                # Size proportional to log2(pixel_count), min 2, max 6
                px = isl['pixel_count']
                sz = min(6, max(2, int(px.bit_length()))) if px > 0 else 2
                pygame.draw.polygon(screen, color, [
                    (ix, iy - sz), (ix + sz, iy), (ix, iy + sz), (ix - sz, iy)
                ])

        # Label with step number (if rect large enough)
        if rw > 30 and rh > 14:
            label = f"{i}"
            label_surf = pygame.font.SysFont("monospace", 10).render(label, True, color)
            screen.blit(label_surf, (rx + 2, ry + 1))

    # Draw final rect (the leaf) — only when showing all steps or at last step
    if focus_step is None or focus_step == len(steps) - 1:
        x1, y1 = state.complex_to_screen(final_rect.re_min_f64(), final_rect.im_min_f64())
        x2, y2 = state.complex_to_screen(final_rect.re_max_f64(), final_rect.im_max_f64())
        rx = min(x1, x2)
        ry = min(y1, y2)
        rw = abs(x2 - x1)
        rh = abs(y2 - y1)
        leaf_rect = pygame.Rect(rx, ry, max(rw, 1), max(rh, 1))
        if leaf_rect.colliderect(vp_clip):
            pygame.draw.rect(screen, point_color, leaf_rect, 3)


def draw_panel(screen, state, font, small_font):
    """Draw the bottom control panel."""
    panel_y = state.vp_h
    panel_rect = pygame.Rect(0, panel_y, state.win_w, state.panel_h)
    pygame.draw.rect(screen, (30, 30, 40), panel_rect)
    pygame.draw.line(screen, (80, 80, 100), (0, panel_y), (state.win_w, panel_y), 1)

    x, y = 10, panel_y + 6

    # Row 1: BIP39 input (debug mode only) or decoded mnemonic display
    if state.debug_mode:
        label = font.render("BIP39:", True, (180, 180, 200))
        screen.blit(label, (x, y))

        input_x = x + label.get_width() + 8
        input_w = state.win_w - input_x - 100
        input_rect = pygame.Rect(input_x, y - 2, input_w, 22)
        border_color = (100, 180, 255) if state.input_focused else (80, 80, 100)
        pygame.draw.rect(screen, (20, 20, 30), input_rect)
        pygame.draw.rect(screen, border_color, input_rect, 1)

        text = state.input_text
        cur = state.input_cursor
        sel = state.input_sel
        sel_lo, sel_hi = min(cur, sel), max(cur, sel)

        txt_surface = small_font.render(text or " ", True, (220, 220, 240))
        cur_px = small_font.size(text[:cur])[0]
        view_w = input_w - 4

        scroll = 0
        if cur_px > view_w:
            scroll = cur_px - view_w + 2

        if state.input_focused and sel_lo != sel_hi:
            sel_lo_px = small_font.size(text[:sel_lo])[0] - scroll
            sel_hi_px = small_font.size(text[:sel_hi])[0] - scroll
            hl_x = max(0, sel_lo_px)
            hl_w = min(view_w, sel_hi_px) - hl_x
            if hl_w > 0:
                hl_surf = pygame.Surface((hl_w, 18))
                hl_surf.fill((60, 100, 180))
                screen.blit(hl_surf, (input_x + 2 + hl_x, y))

        clip_rect = pygame.Rect(scroll, 0, view_w, 20)
        screen.blit(txt_surface, (input_x + 2, y), clip_rect)

        if state.input_focused:
            blink = (pygame.time.get_ticks() // CURSOR_BLINK_MS) % 2 == 0
            if blink:
                cx = input_x + 2 + cur_px - scroll
                pygame.draw.line(screen, (220, 220, 240), (cx, y), (cx, y + 16), 1)

        # Encode button (debug only)
        btn_x = input_x + input_w + 8
        btn_rect = pygame.Rect(btn_x, y - 2, 75, 22)
        btn_color = (60, 120, 60) if not state.select_mode else (60, 60, 80)
        pygame.draw.rect(screen, btn_color, btn_rect)
        pygame.draw.rect(screen, (100, 200, 100), btn_rect, 1)
        btn_txt = font.render("Encode", True, (200, 255, 200))
        screen.blit(btn_txt, (btn_x + 6, y))
        state._encode_btn_rect = btn_rect
    else:
        # Non-debug: show decoded mnemonic (masked) + copy button
        state._encode_btn_rect = pygame.Rect(-100, -100, 0, 0)  # off-screen sentinel
        if state.decoded_mnemonic:
            dec_label = small_font.render("Decoded:", True, (150, 150, 170))
            screen.blit(dec_label, (x, y + 2))
            sx = x + dec_label.get_width() + 8
            # Show word count but not the words themselves
            word_count = len(state.decoded_mnemonic.strip().split())
            if word_count == 12:
                hint = f"{word_count}-word BIP39 mnemonic ready"
                hint_color = (100, 255, 100)
            else:
                hint = state.decoded_mnemonic[:60]
                hint_color = (200, 200, 100)
            hint_surf = small_font.render(hint, True, hint_color)
            screen.blit(hint_surf, (sx, y + 2))
            sx += hint_surf.get_width() + 10
            # Copy button
            copy_rect = pygame.Rect(sx, y, 80, 20)
            pygame.draw.rect(screen, (50, 90, 130), copy_rect)
            pygame.draw.rect(screen, (180, 220, 255), copy_rect, 1)
            copy_txt = small_font.render("Copy", True, (180, 220, 255))
            screen.blit(copy_txt, (sx + 6, y + 2))
            state._copy_mnemonic_btn_rect = copy_rect
            sx += 86

            # Salt input + SHA512 button
            salt_lbl = small_font.render("Salt:", True, (150, 150, 170))
            screen.blit(salt_lbl, (sx, y + 2))
            sx += salt_lbl.get_width() + 4
            salt_fw = 120
            salt_rect = pygame.Rect(sx, y, salt_fw, 20)
            salt_border = (100, 180, 255) if state.salt_focused else (80, 80, 100)
            pygame.draw.rect(screen, (20, 20, 30), salt_rect)
            pygame.draw.rect(screen, salt_border, salt_rect, 1)
            salt_txt = small_font.render(state.salt_text or "", True, (220, 220, 240))
            screen.blit(salt_txt, (sx + 4, y + 2))
            if state.salt_focused:
                blink = (pygame.time.get_ticks() // CURSOR_BLINK_MS) % 2 == 0
                if blink:
                    cx = sx + 4 + small_font.size(state.salt_text[:state.salt_cursor])[0]
                    pygame.draw.line(screen, (220, 220, 240), (cx, y + 2), (cx, y + 16), 1)
            state._salt_rect = salt_rect
            sx += salt_fw + 4
            sha_btn = pygame.Rect(sx, y, 56, 20)
            pygame.draw.rect(screen, (50, 90, 130), sha_btn)
            pygame.draw.rect(screen, (180, 220, 255), sha_btn, 1)
            sha_txt = small_font.render("SHA512", True, (180, 220, 255))
            screen.blit(sha_txt, (sx + 4, y + 2))
            state._sha512_btn_rect = sha_btn
        else:
            info = small_font.render("Select points to decode fractal → BIP39", True, (120, 120, 140))
            screen.blit(info, (x, y + 2))
            state._copy_mnemonic_btn_rect = pygame.Rect(-100, -100, 0, 0)
            state._salt_rect = pygame.Rect(-100, -100, 0, 0)
            state._sha512_btn_rect = pygame.Rect(-100, -100, 0, 0)

    # Row 2: Status + color scheme + mode
    y += 28
    # Color scheme buttons
    scheme_label = small_font.render("Scheme:", True, (150, 150, 170))
    screen.blit(scheme_label, (x, y + 2))
    sx = x + scheme_label.get_width() + 6
    state._scheme_rects = []
    for i, name in enumerate(PALETTE_NAMES):
        active = (i == state.palette_idx)
        btn_r = pygame.Rect(sx, y, 65, 20)
        bg = (60, 80, 120) if active else (40, 40, 55)
        pygame.draw.rect(screen, bg, btn_r)
        pygame.draw.rect(screen, (100, 130, 180) if active else (70, 70, 90), btn_r, 1)
        txt = small_font.render(name, True, (220, 220, 240) if active else (140, 140, 160))
        screen.blit(txt, (sx + 4, y + 2))
        state._scheme_rects.append(btn_r)
        sx += 70

    # Escape-count transform dropdown (click or P to cycle)
    sx += 10
    tf_label = small_font.render("f(esc):", True, (150, 150, 170))
    screen.blit(tf_label, (sx, y + 2))
    sx += tf_label.get_width() + 4
    tf_name = ESC_TRANSFORM_NAMES[state.esc_transform_idx]
    tf_rect = pygame.Rect(sx, y, 75, 20)
    tf_active = state.esc_transform_idx != 0
    tf_bg = (60, 80, 120) if tf_active else (40, 40, 55)
    pygame.draw.rect(screen, tf_bg, tf_rect)
    pygame.draw.rect(screen, (100, 130, 180) if tf_active else (70, 70, 90), tf_rect, 1)
    tf_txt = small_font.render(tf_name, True, (220, 220, 240) if tf_active else (140, 140, 160))
    screen.blit(tf_txt, (sx + 4, y + 2))
    state._esc_transform_rect = tf_rect
    sx += 80

    # Select mode toggle
    sx += 20
    sel_rect = pygame.Rect(sx, y, 90, 20)
    sel_bg = (120, 60, 60) if state.select_mode else (40, 40, 55)
    pygame.draw.rect(screen, sel_bg, sel_rect)
    pygame.draw.rect(screen, (200, 100, 100) if state.select_mode else (70, 70, 90), sel_rect, 1)
    if state.select_mode:
        if state.stage == 2:
            sel_label = f"Select {len(state.stage2_selected_points)}/{state.points_per_stage}"
        else:
            sel_label = f"Select {len(state.selected_points)}/{state.points_per_stage}"
    else:
        sel_label = "Select Pts"
    sel_txt = small_font.render(
        sel_label,
        True, (255, 200, 200) if state.select_mode else (140, 140, 160)
    )
    screen.blit(sel_txt, (sx + 4, y + 2))
    state._select_btn_rect = sel_rect

    # Clear button
    sx += 96
    clr_rect = pygame.Rect(sx, y, 50, 20)
    pygame.draw.rect(screen, (60, 40, 40), clr_rect)
    pygame.draw.rect(screen, (120, 80, 80), clr_rect, 1)
    clr_txt = small_font.render("Clear", True, (200, 150, 150))
    screen.blit(clr_txt, (sx + 4, y + 2))
    state._clear_btn_rect = clr_rect

    # Zoom display
    sx += 60
    zoom_txt = small_font.render(f"Zoom: {state.zoom:.1f}x", True, (150, 150, 170))
    screen.blit(zoom_txt, (sx, y + 2))
    sx += zoom_txt.get_width() + 16

    # Max-iter text input field
    mi_label = small_font.render("Iter:", True, (150, 150, 170))
    screen.blit(mi_label, (sx, y + 2))
    sx += mi_label.get_width() + 4
    mi_w = 55
    mi_rect = pygame.Rect(sx, y, mi_w, 20)
    mi_border = (100, 180, 255) if state.maxiter_focused else (80, 80, 100)
    pygame.draw.rect(screen, (20, 20, 30), mi_rect)
    pygame.draw.rect(screen, mi_border, mi_rect, 1)
    mi_txt = small_font.render(state.maxiter_text, True, (220, 220, 240))
    screen.blit(mi_txt, (sx + 4, y + 2))
    if state.maxiter_focused:
        blink = (pygame.time.get_ticks() // CURSOR_BLINK_MS) % 2 == 0
        if blink:
            cx = sx + 4 + small_font.size(state.maxiter_text[:state.maxiter_cursor])[0]
            pygame.draw.line(screen, (220, 220, 240), (cx, y + 2), (cx, y + 16), 1)
    state._maxiter_rect = mi_rect

    # Row 3: Status message + decoded mnemonic
    y += 24
    status = font.render(state.status_msg, True, state.status_color)
    screen.blit(status, (x, y))

    if state.decoded_mnemonic and state.debug_mode:
        y += 20
        dec_label = small_font.render(f"Decoded: {state.decoded_mnemonic}", True, (100, 255, 100))
        screen.blit(dec_label, (x, y))
        # Copy button in debug mode too
        sx = x + dec_label.get_width() + 10
        copy_rect = pygame.Rect(sx, y, 80, 20)
        pygame.draw.rect(screen, (50, 90, 130), copy_rect)
        pygame.draw.rect(screen, (180, 220, 255), copy_rect, 1)
        copy_txt = small_font.render("Copy", True, (180, 220, 255))
        screen.blit(copy_txt, (sx + 6, y + 2))
        state._copy_mnemonic_btn_rect = copy_rect

    # Row 4: Argon2 — [hex input if debug] | iters field | Hash button | progress/digest
    y += 24
    sx = x

    # Show hex input only in debug mode
    if state.debug_mode:
        if state.argon2_stage1_bits is not None:
            input_hex = bits_to_hex(state.argon2_stage1_bits)
        else:
            input_hex = "—"
        a2_label = small_font.render(f"Argon2 in: {input_hex}", True, (150, 150, 170))
        screen.blit(a2_label, (sx, y + 2))
        sx += a2_label.get_width() + 10
    else:
        a2_label = small_font.render("Argon2", True, (150, 150, 170))
        screen.blit(a2_label, (sx, y + 2))
        sx += a2_label.get_width() + 10

    # Profile toggle button (Basic / Advanced / Great Wall)
    _PROF_DISPLAY = {
        PROFILE_BASIC:      ("Basic",      (50, 80, 50),  (180, 255, 180)),
        PROFILE_ADVANCED:   ("Advanced",   (80, 60, 30),  (255, 210, 140)),
        PROFILE_GREAT_WALL: ("Great Wall", (100, 50, 50), (255, 180, 180)),
    }
    prof_text, prof_bg, prof_fg = _PROF_DISPLAY.get(
        state.argon2_profile, _PROF_DISPLAY[PROFILE_BASIC])
    prof_w = small_font.size(prof_text)[0] + 12
    prof_rect = pygame.Rect(sx, y, prof_w, 20)
    pygame.draw.rect(screen, prof_bg, prof_rect)
    pygame.draw.rect(screen, prof_fg, prof_rect, 1)
    prof_txt = small_font.render(prof_text, True, prof_fg)
    screen.blit(prof_txt, (sx + 6, y + 2))
    state._argon2_profile_rect = prof_rect
    sx += prof_w + 6

    # Iterations input field
    iter_label = small_font.render("iters:", True, (150, 150, 170))
    screen.blit(iter_label, (sx, y + 2))
    sx += iter_label.get_width() + 4
    iter_w = 55
    iter_rect = pygame.Rect(sx, y, iter_w, 20)
    iter_border = (100, 180, 255) if state.argon2_iter_focused else (80, 80, 100)
    pygame.draw.rect(screen, (20, 20, 30), iter_rect)
    pygame.draw.rect(screen, iter_border, iter_rect, 1)
    iter_txt = small_font.render(state.argon2_iterations, True, (220, 220, 240))
    screen.blit(iter_txt, (sx + 4, y + 2))
    if state.argon2_iter_focused:
        blink = (pygame.time.get_ticks() // CURSOR_BLINK_MS) % 2 == 0
        if blink:
            cx = sx + 4 + small_font.size(state.argon2_iterations[:state.argon2_iter_cursor])[0]
            pygame.draw.line(screen, (220, 220, 240), (cx, y + 2), (cx, y + 16), 1)
    state._argon2_iter_rect = iter_rect
    sx += iter_w + 6

    # Hash button
    has_bits = state.argon2_stage1_bits is not None
    if state.argon2_running:
        hash_bg = (80, 80, 40)
        hash_fg = (200, 200, 100)
        hash_label = "Stop"
    elif has_bits:
        hash_bg = (50, 90, 130)
        hash_fg = (180, 220, 255)
        hash_label = "Hash"
    else:
        hash_bg = (40, 40, 50)
        hash_fg = (100, 100, 120)
        hash_label = "Hash"
    hash_w = small_font.size(hash_label)[0] + 12
    hash_btn = pygame.Rect(sx, y, hash_w, 20)
    pygame.draw.rect(screen, hash_bg, hash_btn)
    pygame.draw.rect(screen, hash_fg, hash_btn, 1)
    hash_txt = small_font.render(hash_label, True, hash_fg)
    screen.blit(hash_txt, (sx + 6, y + 2))
    state._argon2_hash_btn_rect = hash_btn
    sx += hash_w + 8

    # Progress bar or digest
    if state.argon2_running and state.argon2_progress_total > 0:
        # Progress bar
        bar_w = state.win_w - sx - x
        bar_h = 16
        bar_y = y + 2
        # Background
        pygame.draw.rect(screen, (30, 30, 45), (sx, bar_y, bar_w, bar_h))
        pygame.draw.rect(screen, (70, 70, 90), (sx, bar_y, bar_w, bar_h), 1)
        # Fill
        frac = state.argon2_progress / state.argon2_progress_total
        fill_w = max(0, int((bar_w - 2) * frac))
        if fill_w > 0:
            pygame.draw.rect(screen, (60, 140, 220), (sx + 1, bar_y + 1, fill_w, bar_h - 2))
        # Label
        pct_text = f"{state.argon2_progress}/{state.argon2_progress_total}"
        pct_surf = small_font.render(pct_text, True, (220, 220, 240))
        pct_x = sx + (bar_w - pct_surf.get_width()) // 2
        screen.blit(pct_surf, (pct_x, bar_y))
    elif state.argon2_digest and state.debug_mode:
        dig_label = small_font.render(f"out: {state.argon2_digest}", True, (100, 220, 255))
        screen.blit(dig_label, (sx, y + 2))

    # Stage indicator and stage-2 parameters
    y += 22
    if state.stage == 2 and state.stage2_p is not None:
        if state.debug_mode:
            o_str = ""
            if state.stage2_o is not None:
                o_str = f"  o=0x{state.stage2_o:016X}  Re(o)={state.stage2_o_re:.8f}  Im(o)={state.stage2_o_im:.8f}"
            s2_txt = small_font.render(
                f"[Stage 2 ACTIVE] p=0x{state.stage2_p:016X}  Re(p)={state.stage2_p_re:.8f}  Im(p)={state.stage2_p_im:.8f}{o_str}",
                True, (140, 255, 140),
            )
        else:
            s2_txt = small_font.render(
                "[Stage 2 ACTIVE]",
                True, (140, 255, 140),
            )
    elif state.stage2_p is not None:
        if state.debug_mode:
            s2_txt = small_font.render(
                f"[Stage 1] (Stage 2 ready: Re(p)={state.stage2_p_re:.6f} Re(o)={state.stage2_o_re:.6f}, press T)",
                True, (180, 220, 140),
            )
        else:
            s2_txt = small_font.render(
                "[Stage 1] (Stage 2 ready, press T)",
                True, (180, 220, 140),
            )
    else:
        preset_label = f"{state.size_preset} ({state.bip39_words}w/{state.entropy_bits}b)"
        s2_txt = small_font.render(
            f"[Stage 1] {preset_label}  (W to cycle)",
            True, (180, 180, 180),
        )
    screen.blit(s2_txt, (x, y + 2))

    # Debug row: manual hex perturbation inputs + "Go" button
    if state.debug_mode:
        y += 22
        sx = x
        lbl = small_font.render("p hex  Re:", True, (150, 150, 170))
        screen.blit(lbl, (sx, y + 2))
        sx += lbl.get_width() + 4

        # Re(p) hex input field
        re_fw = small_font.size("0" * DEBUG_HEX_FIELD_CHARS)[0] + 10
        re_rect = pygame.Rect(sx, y, re_fw, 20)
        re_border = (100, 180, 255) if state.debug_p_re_focused else (80, 80, 100)
        pygame.draw.rect(screen, (20, 20, 30), re_rect)
        pygame.draw.rect(screen, re_border, re_rect, 1)
        re_txt = small_font.render(state.debug_p_re_hex, True, (220, 220, 240))
        screen.blit(re_txt, (sx + 4, y + 2))
        if state.debug_p_re_focused:
            blink = (pygame.time.get_ticks() // CURSOR_BLINK_MS) % 2 == 0
            if blink:
                cx = sx + 4 + small_font.size(state.debug_p_re_hex[:state.debug_p_re_cursor])[0]
                pygame.draw.line(screen, (220, 220, 240), (cx, y + 2), (cx, y + 16), 1)
        state._debug_p_re_rect = re_rect
        sx += re_fw + 8

        lbl2 = small_font.render("Im:", True, (150, 150, 170))
        screen.blit(lbl2, (sx, y + 2))
        sx += lbl2.get_width() + 4

        # Im(p) hex input field
        im_fw = small_font.size("0" * DEBUG_HEX_FIELD_CHARS)[0] + 10
        im_rect = pygame.Rect(sx, y, im_fw, 20)
        im_border = (100, 180, 255) if state.debug_p_im_focused else (80, 80, 100)
        pygame.draw.rect(screen, (20, 20, 30), im_rect)
        pygame.draw.rect(screen, im_border, im_rect, 1)
        im_txt = small_font.render(state.debug_p_im_hex, True, (220, 220, 240))
        screen.blit(im_txt, (sx + 4, y + 2))
        if state.debug_p_im_focused:
            blink = (pygame.time.get_ticks() // CURSOR_BLINK_MS) % 2 == 0
            if blink:
                cx = sx + 4 + small_font.size(state.debug_p_im_hex[:state.debug_p_im_cursor])[0]
                pygame.draw.line(screen, (220, 220, 240), (cx, y + 2), (cx, y + 16), 1)
        state._debug_p_im_rect = im_rect
        sx += im_fw + 8

        # q (linear perturbation) hex fields on next row
        y += 22
        sx = x
        lbl_q = small_font.render("q hex  Re:", True, (150, 150, 170))
        screen.blit(lbl_q, (sx, y + 2))
        sx += lbl_q.get_width() + 4

        q_re_fw = small_font.size("0" * DEBUG_HEX_FIELD_CHARS)[0] + 10
        q_re_rect = pygame.Rect(sx, y, q_re_fw, 20)
        q_re_border = (100, 180, 255) if state.debug_q_re_focused else (80, 80, 100)
        pygame.draw.rect(screen, (20, 20, 30), q_re_rect)
        pygame.draw.rect(screen, q_re_border, q_re_rect, 1)
        q_re_txt = small_font.render(state.debug_q_re_hex, True, (220, 220, 240))
        screen.blit(q_re_txt, (sx + 4, y + 2))
        if state.debug_q_re_focused:
            blink = (pygame.time.get_ticks() // CURSOR_BLINK_MS) % 2 == 0
            if blink:
                cx = sx + 4 + small_font.size(state.debug_q_re_hex[:state.debug_q_re_cursor])[0]
                pygame.draw.line(screen, (220, 220, 240), (cx, y + 2), (cx, y + 16), 1)
        state._debug_q_re_rect = q_re_rect
        sx += q_re_fw + 8

        lbl_q2 = small_font.render("Im:", True, (150, 150, 170))
        screen.blit(lbl_q2, (sx, y + 2))
        sx += lbl_q2.get_width() + 4

        q_im_fw = small_font.size("0" * DEBUG_HEX_FIELD_CHARS)[0] + 10
        q_im_rect = pygame.Rect(sx, y, q_im_fw, 20)
        q_im_border = (100, 180, 255) if state.debug_q_im_focused else (80, 80, 100)
        pygame.draw.rect(screen, (20, 20, 30), q_im_rect)
        pygame.draw.rect(screen, q_im_border, q_im_rect, 1)
        q_im_txt = small_font.render(state.debug_q_im_hex, True, (220, 220, 240))
        screen.blit(q_im_txt, (sx + 4, y + 2))
        if state.debug_q_im_focused:
            blink = (pygame.time.get_ticks() // CURSOR_BLINK_MS) % 2 == 0
            if blink:
                cx = sx + 4 + small_font.size(state.debug_q_im_hex[:state.debug_q_im_cursor])[0]
                pygame.draw.line(screen, (220, 220, 240), (cx, y + 2), (cx, y + 16), 1)
        state._debug_q_im_rect = q_im_rect
        sx += q_im_fw + 8

        # o (orbit seed) hex fields on next row
        y += 22
        sx = x
        lbl_o = small_font.render("o hex  Re:", True, (150, 150, 170))
        screen.blit(lbl_o, (sx, y + 2))
        sx += lbl_o.get_width() + 4

        o_re_fw = small_font.size("0" * DEBUG_HEX_FIELD_CHARS)[0] + 10
        o_re_rect = pygame.Rect(sx, y, o_re_fw, 20)
        o_re_border = (100, 180, 255) if state.debug_o_re_focused else (80, 80, 100)
        pygame.draw.rect(screen, (20, 20, 30), o_re_rect)
        pygame.draw.rect(screen, o_re_border, o_re_rect, 1)
        o_re_txt = small_font.render(state.debug_o_re_hex, True, (220, 220, 240))
        screen.blit(o_re_txt, (sx + 4, y + 2))
        if state.debug_o_re_focused:
            blink = (pygame.time.get_ticks() // CURSOR_BLINK_MS) % 2 == 0
            if blink:
                cx = sx + 4 + small_font.size(state.debug_o_re_hex[:state.debug_o_re_cursor])[0]
                pygame.draw.line(screen, (220, 220, 240), (cx, y + 2), (cx, y + 16), 1)
        state._debug_o_re_rect = o_re_rect
        sx += o_re_fw + 8

        lbl_o2 = small_font.render("Im:", True, (150, 150, 170))
        screen.blit(lbl_o2, (sx, y + 2))
        sx += lbl_o2.get_width() + 4

        o_im_fw = small_font.size("0" * DEBUG_HEX_FIELD_CHARS)[0] + 10
        o_im_rect = pygame.Rect(sx, y, o_im_fw, 20)
        o_im_border = (100, 180, 255) if state.debug_o_im_focused else (80, 80, 100)
        pygame.draw.rect(screen, (20, 20, 30), o_im_rect)
        pygame.draw.rect(screen, o_im_border, o_im_rect, 1)
        o_im_txt = small_font.render(state.debug_o_im_hex, True, (220, 220, 240))
        screen.blit(o_im_txt, (sx + 4, y + 2))
        if state.debug_o_im_focused:
            blink = (pygame.time.get_ticks() // CURSOR_BLINK_MS) % 2 == 0
            if blink:
                cx = sx + 4 + small_font.size(state.debug_o_im_hex[:state.debug_o_im_cursor])[0]
                pygame.draw.line(screen, (220, 220, 240), (cx, y + 2), (cx, y + 16), 1)
        state._debug_o_im_rect = o_im_rect
        sx += o_im_fw + 8

        # "Go" button — apply manual hex p,q and switch to stage 2
        go_btn = pygame.Rect(sx, y, 36, 20)
        pygame.draw.rect(screen, (50, 90, 130), go_btn)
        pygame.draw.rect(screen, (180, 220, 255), go_btn, 1)
        go_txt = small_font.render("Go", True, (180, 220, 255))
        screen.blit(go_txt, (sx + 8, y + 2))
        state._debug_go_btn_rect = go_btn


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Burning Ship fractal viewer")
    parser.add_argument("--cache-size", type=int, default=0,
                        help="Render cache capacity (number of entries). "
                             "0 = disabled. Recommended: 1048576 (2^20) or higher.")
    args = parser.parse_args()

    # Prevent Ctrl+C (SIGINT) from generating a pygame.QUIT event;
    # Ctrl+C is used as copy in text fields.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    pygame.init()
    pygame.scrap.init()

    if args.cache_size > 0:
        cache_init(args.cache_size)
        cache_init_stage2(args.cache_size)

    state = ViewerState()
    screen = pygame.display.set_mode((state.win_w, state.win_h), pygame.RESIZABLE)
    pygame.display.set_caption("Burning Ship — BIP39 Encoder")

    font = pygame.font.SysFont("monospace", 14)
    small_font = pygame.font.SysFont("monospace", 12)

    clock = pygame.time.Clock()
    dragging = False
    drag_start = None
    fractal_surface = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                state.win_w = event.w
                state.win_h = event.h
                state.vp_w = event.w
                state.vp_h = event.h - state.panel_h
                state.pixel_buf = np.zeros(state.vp_w * state.vp_h, dtype=np.uint8)
                screen = pygame.display.set_mode((state.win_w, state.win_h), pygame.RESIZABLE)
                state.needs_redraw = True

            elif event.type == pygame.KEYDOWN:
                if state.input_focused:
                    handle_text_input(state, event)
                elif state.argon2_iter_focused:
                    # Numeric-only editing for iterations field
                    if event.key in (pygame.K_ESCAPE, pygame.K_TAB, pygame.K_RETURN):
                        state.argon2_iter_focused = False
                    elif event.key == pygame.K_BACKSPACE:
                        if state.argon2_iter_cursor > 0:
                            t = state.argon2_iterations
                            c = state.argon2_iter_cursor
                            state.argon2_iterations = t[:c-1] + t[c:]
                            state.argon2_iter_cursor = c - 1
                    elif event.key == pygame.K_DELETE:
                        t = state.argon2_iterations
                        c = state.argon2_iter_cursor
                        if c < len(t):
                            state.argon2_iterations = t[:c] + t[c+1:]
                    elif event.key == pygame.K_LEFT:
                        state.argon2_iter_cursor = max(0, state.argon2_iter_cursor - 1)
                    elif event.key == pygame.K_RIGHT:
                        state.argon2_iter_cursor = min(len(state.argon2_iterations), state.argon2_iter_cursor + 1)
                    elif event.key == pygame.K_HOME:
                        state.argon2_iter_cursor = 0
                    elif event.key == pygame.K_END:
                        state.argon2_iter_cursor = len(state.argon2_iterations)
                    elif event.unicode and event.unicode.isdigit():
                        t = state.argon2_iterations
                        c = state.argon2_iter_cursor
                        state.argon2_iterations = t[:c] + event.unicode + t[c:]
                        state.argon2_iter_cursor = c + 1
                elif state.maxiter_focused:
                    # Numeric-only editing for max-iter field
                    if event.key in (pygame.K_ESCAPE, pygame.K_TAB, pygame.K_RETURN):
                        state.maxiter_focused = False
                        # Apply the value
                        try:
                            val = int(state.maxiter_text)
                            val = max(MAX_ITER_MIN, min(MAX_ITER_MAX, val))
                        except ValueError:
                            val = DEFAULT_MAX_ITER
                        if val != state.max_iter:
                            state.max_iter = val
                            cache_clear()
                            cache_clear_stage2()
                            state.needs_redraw = True
                        state.maxiter_text = str(state.max_iter)
                        state.maxiter_cursor = len(state.maxiter_text)
                    elif event.key == pygame.K_BACKSPACE:
                        if state.maxiter_cursor > 0:
                            t = state.maxiter_text
                            c = state.maxiter_cursor
                            state.maxiter_text = t[:c-1] + t[c:]
                            state.maxiter_cursor = c - 1
                    elif event.key == pygame.K_DELETE:
                        t = state.maxiter_text
                        c = state.maxiter_cursor
                        if c < len(t):
                            state.maxiter_text = t[:c] + t[c+1:]
                    elif event.key == pygame.K_LEFT:
                        state.maxiter_cursor = max(0, state.maxiter_cursor - 1)
                    elif event.key == pygame.K_RIGHT:
                        state.maxiter_cursor = min(len(state.maxiter_text), state.maxiter_cursor + 1)
                    elif event.key == pygame.K_HOME:
                        state.maxiter_cursor = 0
                    elif event.key == pygame.K_END:
                        state.maxiter_cursor = len(state.maxiter_text)
                    elif event.unicode and event.unicode.isdigit():
                        t = state.maxiter_text
                        c = state.maxiter_cursor
                        state.maxiter_text = t[:c] + event.unicode + t[c:]
                        state.maxiter_cursor = c + 1
                elif state.salt_focused:
                    # Free-text editing for salt field
                    if event.key in (pygame.K_ESCAPE, pygame.K_TAB, pygame.K_RETURN):
                        state.salt_focused = False
                    elif event.key == pygame.K_BACKSPACE:
                        if state.salt_cursor > 0:
                            t = state.salt_text
                            c = state.salt_cursor
                            state.salt_text = t[:c-1] + t[c:]
                            state.salt_cursor = c - 1
                    elif event.key == pygame.K_DELETE:
                        t = state.salt_text
                        c = state.salt_cursor
                        if c < len(t):
                            state.salt_text = t[:c] + t[c+1:]
                    elif event.key == pygame.K_LEFT:
                        state.salt_cursor = max(0, state.salt_cursor - 1)
                    elif event.key == pygame.K_RIGHT:
                        state.salt_cursor = min(len(state.salt_text), state.salt_cursor + 1)
                    elif event.key == pygame.K_HOME:
                        state.salt_cursor = 0
                    elif event.key == pygame.K_END:
                        state.salt_cursor = len(state.salt_text)
                    elif event.unicode and event.unicode.isprintable():
                        t = state.salt_text
                        c = state.salt_cursor
                        state.salt_text = t[:c] + event.unicode + t[c:]
                        state.salt_cursor = c + 1
                elif state.debug_p_re_focused or state.debug_p_im_focused:
                    # Hex-only editing for debug perturbation fields
                    editing_re = state.debug_p_re_focused
                    if editing_re:
                        field, cursor = state.debug_p_re_hex, state.debug_p_re_cursor
                    else:
                        field, cursor = state.debug_p_im_hex, state.debug_p_im_cursor
                    if event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                        state.debug_p_re_focused = False
                        state.debug_p_im_focused = False
                    elif event.key == pygame.K_TAB:
                        # Tab toggles between Re and Im fields
                        if editing_re:
                            state.debug_p_re_focused = False
                            state.debug_p_im_focused = True
                        else:
                            state.debug_p_im_focused = False
                            state.debug_p_re_focused = True
                    elif event.key == pygame.K_BACKSPACE:
                        if cursor > 0:
                            field = field[:cursor-1] + field[cursor:]
                            cursor -= 1
                    elif event.key == pygame.K_DELETE:
                        if cursor < len(field):
                            field = field[:cursor] + field[cursor+1:]
                    elif event.key == pygame.K_LEFT:
                        cursor = max(0, cursor - 1)
                    elif event.key == pygame.K_RIGHT:
                        cursor = min(len(field), cursor + 1)
                    elif event.key == pygame.K_HOME:
                        cursor = 0
                    elif event.key == pygame.K_END:
                        cursor = len(field)
                    elif event.unicode and event.unicode.lower() in '0123456789abcdef' and len(field) < DEBUG_HEX_FIELD_CHARS:
                        field = field[:cursor] + event.unicode.upper() + field[cursor:]
                        cursor += 1
                    # Write back
                    if editing_re:
                        state.debug_p_re_hex, state.debug_p_re_cursor = field, cursor
                    else:
                        state.debug_p_im_hex, state.debug_p_im_cursor = field, cursor
                elif state.debug_q_re_focused or state.debug_q_im_focused:
                    # Hex-only editing for debug linear perturbation (q) fields
                    editing_re = state.debug_q_re_focused
                    if editing_re:
                        field, cursor = state.debug_q_re_hex, state.debug_q_re_cursor
                    else:
                        field, cursor = state.debug_q_im_hex, state.debug_q_im_cursor
                    if event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                        state.debug_q_re_focused = False
                        state.debug_q_im_focused = False
                    elif event.key == pygame.K_TAB:
                        if editing_re:
                            state.debug_q_re_focused = False
                            state.debug_q_im_focused = True
                        else:
                            state.debug_q_im_focused = False
                            state.debug_q_re_focused = True
                    elif event.key == pygame.K_BACKSPACE:
                        if cursor > 0:
                            field = field[:cursor-1] + field[cursor:]
                            cursor -= 1
                    elif event.key == pygame.K_DELETE:
                        if cursor < len(field):
                            field = field[:cursor] + field[cursor+1:]
                    elif event.key == pygame.K_LEFT:
                        cursor = max(0, cursor - 1)
                    elif event.key == pygame.K_RIGHT:
                        cursor = min(len(field), cursor + 1)
                    elif event.key == pygame.K_HOME:
                        cursor = 0
                    elif event.key == pygame.K_END:
                        cursor = len(field)
                    elif event.unicode and event.unicode.lower() in '0123456789abcdef' and len(field) < DEBUG_HEX_FIELD_CHARS:
                        field = field[:cursor] + event.unicode.upper() + field[cursor:]
                        cursor += 1
                    if editing_re:
                        state.debug_q_re_hex, state.debug_q_re_cursor = field, cursor
                    else:
                        state.debug_q_im_hex, state.debug_q_im_cursor = field, cursor
                elif state.debug_o_re_focused or state.debug_o_im_focused:
                    # Hex-only editing for debug orbit seed (o) fields
                    editing_re = state.debug_o_re_focused
                    if editing_re:
                        field, cursor = state.debug_o_re_hex, state.debug_o_re_cursor
                    else:
                        field, cursor = state.debug_o_im_hex, state.debug_o_im_cursor
                    if event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                        state.debug_o_re_focused = False
                        state.debug_o_im_focused = False
                    elif event.key == pygame.K_TAB:
                        if editing_re:
                            state.debug_o_re_focused = False
                            state.debug_o_im_focused = True
                        else:
                            state.debug_o_im_focused = False
                            state.debug_o_re_focused = True
                    elif event.key == pygame.K_BACKSPACE:
                        if cursor > 0:
                            field = field[:cursor-1] + field[cursor:]
                            cursor -= 1
                    elif event.key == pygame.K_DELETE:
                        if cursor < len(field):
                            field = field[:cursor] + field[cursor+1:]
                    elif event.key == pygame.K_LEFT:
                        cursor = max(0, cursor - 1)
                    elif event.key == pygame.K_RIGHT:
                        cursor = min(len(field), cursor + 1)
                    elif event.key == pygame.K_HOME:
                        cursor = 0
                    elif event.key == pygame.K_END:
                        cursor = len(field)
                    elif event.unicode and event.unicode.lower() in '0123456789abcdef' and len(field) < DEBUG_HEX_FIELD_CHARS:
                        field = field[:cursor] + event.unicode.upper() + field[cursor:]
                        cursor += 1
                    if editing_re:
                        state.debug_o_re_hex, state.debug_o_re_cursor = field, cursor
                    else:
                        state.debug_o_im_hex, state.debug_o_im_cursor = field, cursor
                else:
                    if event.key == pygame.K_TAB:
                        if state.debug_mode:
                            state.input_focused = True
                            state.argon2_iter_focused = False
                            state.maxiter_focused = False
                            state.salt_focused = False
                            state.debug_o_re_focused = False
                            state.debug_o_im_focused = False
                            state.debug_p_re_focused = False
                            state.debug_p_im_focused = False
                            state.debug_q_re_focused = False
                            state.debug_q_im_focused = False
                    elif event.key == pygame.K_ESCAPE:
                        if state.stretch_mode:
                            # Abort stretch correction point selection
                            state.stretch_mode = False
                            state.stretch_p1 = None
                            state.status_msg = "Stretch correction cancelled"
                            state.status_color = CLR_NEUTRAL
                        else:
                            running = False
                    elif event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        state.reset_view()
                    elif event.key == pygame.K_c:
                        state.stage1_encoded_points = []
                        state.stage1_encoded_bits_chunks = []
                        state.stage1_encoded_steps = []
                        state.stage1_encoded_final_rects = []
                        state.stage2_encoded_points = []
                        state.stage2_encoded_bits_chunks = []
                        state.stage2_encoded_steps = []
                        state.stage2_encoded_final_rects = []
                        state.selected_point_idx = None
                        state.selected_decoded_idx = None
                        state.selected_points = []
                        state.decoded_mnemonic = ""
                        state.highlighted_leaf_rect = None
                        state.needs_repalette = True
                        state.status_msg = "Cleared"
                        state.status_color = CLR_NEUTRAL
                    elif event.key == pygame.K_n:
                        # Next point (encoded then selected)
                        n_enc = len(state.encoded_points)
                        n_sel = len(state.selected_points)
                        if n_enc + n_sel > 0:
                            if state.selected_point_idx is not None:
                                nxt = state.selected_point_idx + 1
                                if nxt < n_enc:
                                    state.selected_point_idx = nxt
                                elif n_sel > 0:
                                    state.selected_point_idx = None
                                    state.selected_decoded_idx = 0
                                else:
                                    state.selected_point_idx = 0
                            elif state.selected_decoded_idx is not None:
                                nxt = state.selected_decoded_idx + 1
                                if nxt < n_sel:
                                    state.selected_decoded_idx = nxt
                                elif n_enc > 0:
                                    state.selected_decoded_idx = None
                                    state.selected_point_idx = 0
                                else:
                                    state.selected_decoded_idx = 0
                            else:
                                if n_enc > 0:
                                    state.selected_point_idx = 0
                                else:
                                    state.selected_decoded_idx = 0
                            if state.selected_point_idx is not None:
                                point_offset = 0 if state.stage == 1 else state.points_per_stage
                                lbl = f"P{state.selected_point_idx + 1 + point_offset}"
                                state.status_color = MARKER_COLORS[state.selected_point_idx % len(MARKER_COLORS)]
                            else:
                                lbl = f"S{state.selected_decoded_idx + 1}"
                                state.status_color = MARKER_COLORS[state.selected_decoded_idx % len(MARKER_COLORS)]
                            state.status_msg = f"Selected {lbl}"
                    elif event.key == pygame.K_p:
                        # Cycle escape-count transform
                        state.esc_transform_idx = (state.esc_transform_idx + 1) % len(ESC_TRANSFORMS)
                        name = ESC_TRANSFORM_NAMES[state.esc_transform_idx]
                        state.status_msg = f"Esc transform: {name}"
                        state.status_color = CLR_NEUTRAL
                        state.needs_repalette = True
                    elif event.key == pygame.K_d:
                        # Toggle debug mode
                        state.debug_mode = not state.debug_mode
                        if not state.debug_mode:
                            state.input_focused = False
                        label = "ON" if state.debug_mode else "OFF"
                        state.status_msg = f"Debug mode {label}"
                        state.status_color = CLR_NEUTRAL
                    elif event.key == pygame.K_w:
                        # Cycle size preset: mini → default → large → mini
                        idx = SIZE_PRESET_ORDER.index(state.size_preset)
                        state.size_preset = SIZE_PRESET_ORDER[(idx + 1) % len(SIZE_PRESET_ORDER)]
                        state.status_msg = (f"Size: {state.size_preset} "
                            f"({state.bip39_words} words, {state.entropy_bits} bits, "
                            f"{state.points_per_stage} pts/stage)")
                        state.status_color = CLR_NEUTRAL
                    elif event.key == pygame.K_s:
                        state.select_mode = not state.select_mode
                        if state.select_mode:
                            if state.stage == 2:
                                state.stage2_selected_points = []
                                state.status_msg = f"Click {state.points_per_stage} points to decode (stage 2)"
                            else:
                                state.selected_points = []
                                state.status_msg = f"Click {state.points_per_stage} points to decode (stage 1)"
                            state.status_color = CLR_WARNING
                        else:
                            state.status_msg = "Select mode off"
                            state.status_color = CLR_NEUTRAL
                    elif event.key == pygame.K_t:
                        if state.stage == 1 and state.stage2_p is not None:
                            state.stage = 2
                            if state.debug_mode:
                                state.status_msg = (f"Stage 2  Re(p)={state.stage2_p_re:.6f} Im(p)={state.stage2_p_im:.6f}"
                                                    f"  Re(o)={state.stage2_o_re:.6f} Im(o)={state.stage2_o_im:.6f}")
                            else:
                                state.status_msg = "Stage 2"
                            state.status_color = CLR_STAGE_RDY
                            state.needs_redraw = True
                        elif state.stage == 2:
                            state.stage = 1
                            state.status_msg = "Stage 1 (d=2)"
                            state.status_color = CLR_NEUTRAL
                            state.needs_redraw = True
                        else:
                            state.status_msg = "Stage 2 unavailable (run Argon2 first)"
                            state.status_color = CLR_WARNING
                    elif event.key == pygame.K_v:
                        state.show_areas = not state.show_areas
                        state.area_focus_step = None
                        label = "ON" if state.show_areas else "OFF"
                        state.status_msg = f"Area visualization {label}"
                        state.status_color = CLR_NEUTRAL
                    elif event.key == pygame.K_m:
                        state.manual_bits_mode = not state.manual_bits_mode
                        if state.manual_bits_mode:
                            state.manual_bits = []
                            state.manual_point_idx = 0
                            state.manual_encode_result = None
                            state.manual_encode_history = []
                            state.manual_committed_points = []
                            state.manual_committed_steps = []
                            state.manual_committed_rects = []
                            state.manual_committed_bits = []
                            state.manual_path_prefix = "O"
                            state.show_areas = True
                            state.status_msg = (f"Manual P1: O=0 (left/up), I=1 (right/down), "
                                                f"Backspace=undo  [0/{BITS_PER_POINT}]")
                            state.status_color = CLR_BIT_OK
                        else:
                            state.status_msg = "Manual bit input OFF"
                            state.status_color = CLR_NEUTRAL
                    elif event.key == pygame.K_o and state.manual_bits_mode:
                        manual_add_bit(state, 0)
                    elif event.key == pygame.K_i and state.manual_bits_mode:
                        manual_add_bit(state, 1)
                    elif event.key == pygame.K_BACKSPACE and state.manual_bits_mode and not state.input_focused:
                        if state.manual_bits and state.manual_encode_history:
                            state.manual_bits.pop()
                            # Restore cached result (no recomputation)
                            state.manual_encode_result = state.manual_encode_history.pop()
                            manual_update_viz(state)
                            n = len(state.manual_bits)
                            pt = state.manual_point_idx + 1
                            path = state.manual_encode_result.path if state.manual_encode_result else "O"
                            state.status_msg = f"P{pt} bit removed → {n}/{BITS_PER_POINT}, path={path}"
                            state.status_color = CLR_WARNING
                        elif state.manual_point_idx > 0 and state.manual_committed_bits:
                            # Uncommit previous point and pop its last bit
                            prev_bits = state.manual_committed_bits.pop()
                            state.manual_committed_points.pop()
                            state.manual_committed_steps.pop()
                            state.manual_committed_rects.pop()
                            state.manual_point_idx -= 1
                            # Restore bits minus the last one
                            state.manual_bits = prev_bits[:-1]
                            manual_encode_latest(state)
                            n = len(state.manual_bits)
                            pt = state.manual_point_idx + 1
                            path = state.manual_encode_result.path if state.manual_encode_result else "O"
                            state.status_msg = f"Back to P{pt} → {n}/{BITS_PER_POINT}, path={path}"
                            state.status_color = CLR_WARNING
                    elif event.key == pygame.K_COMMA:
                        # Previous step (< key)
                        if state.show_areas:
                            total = _area_total_steps(state)
                            if total > 0:
                                if state.area_focus_step is None:
                                    state.area_focus_step = total - 1
                                elif state.area_focus_step > 0:
                                    state.area_focus_step -= 1
                                state.status_msg = f"Step {state.area_focus_step}/{total-1}"
                                state.status_color = CLR_NEUTRAL
                    elif event.key == pygame.K_PERIOD:
                        # Next step (> key)
                        if state.show_areas:
                            total = _area_total_steps(state)
                            if total > 0:
                                if state.area_focus_step is None:
                                    state.area_focus_step = 0
                                elif state.area_focus_step < total - 1:
                                    state.area_focus_step += 1
                                else:
                                    state.area_focus_step = None
                                    state.status_msg = "Showing all steps"
                                    state.status_color = CLR_NEUTRAL
                                    continue
                                state.status_msg = f"Step {state.area_focus_step}/{total-1}"
                                state.status_color = CLR_NEUTRAL
                    elif event.key == pygame.K_l:
                        state.brightness_exponent_offset += BRIGHTNESS_OFFSET_STEP
                        state.status_msg = f"Brightness offset {state.brightness_exponent_offset:+.1f}"
                        state.status_color = CLR_NEUTRAL
                        state.needs_repalette = True
                    elif event.key == pygame.K_k:
                        state.brightness_exponent_offset -= BRIGHTNESS_OFFSET_STEP
                        state.status_msg = f"Brightness offset {state.brightness_exponent_offset:+.1f}"
                        state.status_color = CLR_NEUTRAL
                        state.needs_repalette = True
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                        idx = event.key - pygame.K_1
                        if idx < len(PALETTE_NAMES):
                            state.palette_idx = idx
                            state.needs_repalette = True
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        state.zoom_exp += 2  # *2 (two sqrt(2) steps)
                        state.needs_redraw = True
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        state.zoom_exp -= 2  # /2 (two sqrt(2) steps)
                        state.needs_redraw = True
                    elif event.key == pygame.K_LEFT:
                        state.center_re -= state.pixel_step() * state.vp_w * 0.25
                        state.needs_redraw = True
                    elif event.key == pygame.K_RIGHT:
                        state.center_re += state.pixel_step() * state.vp_w * 0.25
                        state.needs_redraw = True
                    elif event.key == pygame.K_UP:
                        state.center_im -= state.pixel_step() * state.vp_h * 0.25
                        state.needs_redraw = True
                    elif event.key == pygame.K_DOWN:
                        state.center_im += state.pixel_step() * state.vp_h * 0.25
                        state.needs_redraw = True
                    elif event.key == pygame.K_x:
                        # Enter stretch correction mode: pick P1 then P2
                        state.stretch_mode = True
                        state.stretch_p1 = None
                        state.status_msg = "Stretch correction: click P1"
                        state.status_color = CLR_INFO
                    elif event.key == pygame.K_z:
                        # Clear all stretch corrections
                        if state.stretch_corrections:
                            state.stretch_corrections = []
                            state.needs_repalette = True
                            state.status_msg = "Stretch corrections cleared"
                            state.status_color = CLR_NEUTRAL
                    elif event.key == pygame.K_F2:
                        # Random encode: stage 1 + Argon2 + stage 2
                        if not state.argon2_running:
                            run_random_encode(state)
                    elif event.key == pygame.K_F6:
                        # Save session to JSON
                        if state.stage1_encoded_points:
                            import time as _time
                            ts = _time.strftime("%Y%m%d_%H%M%S")
                            fname = f"bs_session_{ts}.json"
                            try:
                                save_session(state, fname)
                                state.status_msg = f"Saved: {fname}"
                                state.status_color = CLR_SUCCESS
                            except Exception as e:
                                state.status_msg = f"Save error: {e}"
                                state.status_color = CLR_ERROR
                        else:
                            state.status_msg = "Nothing to save (encode first)"
                            state.status_color = CLR_WARNING
                    elif event.key == pygame.K_F5:
                        # Load session from JSON — pick most recent bs_session_*.json
                        import glob as _glob
                        candidates = sorted(_glob.glob("bs_session_*.json"), reverse=True)
                        if candidates:
                            fname = candidates[0]
                            try:
                                load_session(state, fname)
                                state.status_msg = f"Loaded: {fname}"
                                state.status_color = CLR_SUCCESS
                            except Exception as e:
                                state.status_msg = f"Load error: {e}"
                                state.status_color = CLR_ERROR
                        else:
                            state.status_msg = "No bs_session_*.json files found"
                            state.status_color = CLR_WARNING

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # Check panel clicks
                if my >= state.vp_h:
                    if hasattr(state, '_encode_btn_rect') and state._encode_btn_rect.collidepoint(mx, my):
                        try:
                            if state.stage == 2:
                                pts, chunks, steps, frects = encode_bip39_stage2(
                                    state.input_text, state.stage2_o, state.stage2_p,
                                    state.stage2_q, num_points=state.points_per_stage)
                                state.stage2_encoded_points = pts
                                state.stage2_encoded_bits_chunks = chunks
                                state.stage2_encoded_steps = steps
                                state.stage2_encoded_final_rects = frects
                                state.selected_point_idx = None
                                state.selected_decoded_idx = None
                                state.needs_redraw = True
                                state.status_msg = f"Encoded {state.points_per_stage} points ({state.bits_per_stage} bits, stage 2)"
                                state.status_color = CLR_SUCCESS
                            else:
                                pts, chunks, steps, frects = encode_bip39(state.input_text,
                                    num_points=state.points_per_stage)
                                state.stage1_encoded_points = pts
                                state.stage1_encoded_bits_chunks = chunks
                                state.stage1_encoded_steps = steps
                                state.stage1_encoded_final_rects = frects
                                state.selected_point_idx = None
                                state.selected_decoded_idx = None
                                stage1 = []
                                for c in chunks:
                                    stage1.extend(c)
                                state.argon2_stage1_bits = stage1
                                state.argon2_digest = ""
                                state.needs_redraw = True
                                # Auto-chain: if iterations field has a valid number, run Argon2
                                try:
                                    iters = int(state.argon2_iterations)
                                    if iters < 0:
                                        raise ValueError
                                except ValueError:
                                    iters = None
                                if iters is not None and iters >= 0:
                                    state.argon2_running = True
                                    state.argon2_digest = ""
                                    state.argon2_progress = 0
                                    state.argon2_progress_total = max(iters, 1)
                                    prof_label = {PROFILE_BASIC: "Basic", PROFILE_ADVANCED: "Advanced", PROFILE_GREAT_WALL: "Great Wall"}.get(state.argon2_profile, "Basic")
                                    state.status_msg = f"Encoded → Argon2 {prof_label} (x{iters})..."
                                    state.status_color = CLR_PENDING
                                    run_argon2_iterative(state, iters)
                                else:
                                    state.status_msg = f"Encoded {state.points_per_stage} points ({state.bits_per_stage} bits, stage 1)"
                                    state.status_color = CLR_SUCCESS
                        except ValueError as e:
                            state.status_msg = f"Error: {e}"
                            state.status_color = CLR_ERROR
                    elif hasattr(state, '_argon2_profile_rect') and state._argon2_profile_rect.collidepoint(mx, my):
                        if not state.argon2_running:
                            # Cycle: Basic → Advanced → Great Wall → Basic
                            if state.argon2_profile == PROFILE_BASIC:
                                state.argon2_profile = PROFILE_ADVANCED
                                state.status_msg = "Profile: Advanced (32 GiB, p=4, desktop)"
                            elif state.argon2_profile == PROFILE_ADVANCED:
                                state.argon2_profile = PROFILE_GREAT_WALL
                                state.status_msg = "Profile: Great Wall (128 GiB, p=4, server)"
                            else:
                                state.argon2_profile = PROFILE_BASIC
                                state.status_msg = "Profile: Basic (1 GiB, p=2, mobile)"
                            state.status_color = CLR_PENDING
                    elif hasattr(state, '_argon2_hash_btn_rect') and state._argon2_hash_btn_rect.collidepoint(mx, my):
                        if state.argon2_stage1_bits and not state.argon2_running:
                            try:
                                iters = int(state.argon2_iterations)
                                if iters < 0:
                                    raise ValueError("must be >= 0")
                            except ValueError:
                                state.status_msg = "Invalid iteration count (>= 0)"
                                state.status_color = CLR_ERROR
                                continue
                            state.argon2_running = True
                            state.argon2_digest = ""
                            state.argon2_progress = 0
                            state.argon2_progress_total = max(iters, 1)
                            prof_label = {PROFILE_BASIC: "Basic", PROFILE_ADVANCED: "Advanced", PROFILE_GREAT_WALL: "Great Wall"}.get(state.argon2_profile, "Basic")
                            label = "identity" if iters == 0 else f"x{iters}, {prof_label}"
                            state.status_msg = f"Argon2d running ({label})..."
                            state.status_color = CLR_PENDING
                            run_argon2_iterative(state, iters)
                        elif not state.argon2_stage1_bits:
                            state.status_msg = "Encode first to get stage-1 bits"
                            state.status_color = CLR_WARNING
                    elif hasattr(state, '_argon2_iter_rect') and state._argon2_iter_rect.collidepoint(mx, my):
                        state.argon2_iter_focused = True
                        state.input_focused = False
                        state.maxiter_focused = False
                        state.debug_o_re_focused = False
                        state.debug_o_im_focused = False
                        state.debug_p_re_focused = False
                        state.debug_p_im_focused = False
                        state.debug_q_re_focused = False
                        state.debug_q_im_focused = False
                    elif hasattr(state, '_esc_transform_rect') and state._esc_transform_rect.collidepoint(mx, my):
                        state.esc_transform_idx = (state.esc_transform_idx + 1) % len(ESC_TRANSFORMS)
                        name = ESC_TRANSFORM_NAMES[state.esc_transform_idx]
                        state.status_msg = f"Esc transform: {name}"
                        state.status_color = CLR_NEUTRAL
                        state.needs_repalette = True
                    elif hasattr(state, '_select_btn_rect') and state._select_btn_rect.collidepoint(mx, my):
                        state.select_mode = not state.select_mode
                        if state.select_mode:
                            if state.stage == 2:
                                state.stage2_selected_points = []
                                state.status_msg = f"Click {state.points_per_stage} points to decode (stage 2)"
                            else:
                                state.selected_points = []
                                state.status_msg = f"Click {state.points_per_stage} points to decode (stage 1)"
                            state.status_color = CLR_WARNING
                        else:
                            state.status_msg = "Select mode off"
                            state.status_color = CLR_NEUTRAL
                    elif hasattr(state, '_clear_btn_rect') and state._clear_btn_rect.collidepoint(mx, my):
                        state.stage1_encoded_points = []
                        state.stage1_encoded_bits_chunks = []
                        state.stage1_encoded_steps = []
                        state.stage1_encoded_final_rects = []
                        state.stage2_encoded_points = []
                        state.stage2_encoded_bits_chunks = []
                        state.stage2_encoded_steps = []
                        state.stage2_encoded_final_rects = []
                        state.selected_point_idx = None
                        state.selected_decoded_idx = None
                        state.selected_points = []
                        state.selected_steps = []
                        state.selected_final_rects = []
                        state.stage2_selected_points = []
                        state.stage2_selected_steps = []
                        state.stage2_selected_final_rects = []
                        state.decoded_mnemonic = ""
                        state.decoded_stage1_bits = None
                        state.decoded_stage2_bits = None
                        state.area_focus_step = None
                        state.highlighted_leaf_rect = None
                        state.needs_repalette = True
                        state.status_msg = "Cleared"
                        state.status_color = CLR_NEUTRAL
                    elif hasattr(state, '_debug_p_re_rect') and state._debug_p_re_rect.collidepoint(mx, my):
                        state.debug_p_re_focused = True
                        state.debug_o_re_focused = False
                        state.debug_o_im_focused = False
                        state.debug_p_im_focused = False
                        state.debug_q_re_focused = False
                        state.debug_q_im_focused = False
                        state.input_focused = False
                        state.argon2_iter_focused = False
                        state.maxiter_focused = False
                    elif hasattr(state, '_debug_p_im_rect') and state._debug_p_im_rect.collidepoint(mx, my):
                        state.debug_p_im_focused = True
                        state.debug_o_re_focused = False
                        state.debug_o_im_focused = False
                        state.debug_p_re_focused = False
                        state.input_focused = False
                        state.argon2_iter_focused = False
                        state.maxiter_focused = False
                    elif hasattr(state, '_debug_q_re_rect') and state._debug_q_re_rect.collidepoint(mx, my):
                        state.debug_q_re_focused = True
                        state.debug_o_re_focused = False
                        state.debug_o_im_focused = False
                        state.debug_p_re_focused = False
                        state.debug_p_im_focused = False
                        state.debug_q_im_focused = False
                        state.input_focused = False
                        state.argon2_iter_focused = False
                        state.maxiter_focused = False
                    elif hasattr(state, '_debug_q_im_rect') and state._debug_q_im_rect.collidepoint(mx, my):
                        state.debug_q_im_focused = True
                        state.debug_o_re_focused = False
                        state.debug_o_im_focused = False
                        state.debug_p_re_focused = False
                        state.debug_p_im_focused = False
                        state.debug_q_re_focused = False
                        state.input_focused = False
                        state.argon2_iter_focused = False
                        state.maxiter_focused = False
                    elif hasattr(state, '_debug_o_re_rect') and state._debug_o_re_rect.collidepoint(mx, my):
                        state.debug_o_re_focused = True
                        state.debug_o_im_focused = False
                        state.debug_p_re_focused = False
                        state.debug_p_im_focused = False
                        state.debug_q_re_focused = False
                        state.debug_q_im_focused = False
                        state.input_focused = False
                        state.argon2_iter_focused = False
                        state.maxiter_focused = False
                    elif hasattr(state, '_debug_o_im_rect') and state._debug_o_im_rect.collidepoint(mx, my):
                        state.debug_o_im_focused = True
                        state.debug_o_re_focused = False
                        state.debug_p_re_focused = False
                        state.debug_p_im_focused = False
                        state.input_focused = False
                        state.argon2_iter_focused = False
                        state.maxiter_focused = False
                    elif hasattr(state, '_debug_go_btn_rect') and state._debug_go_btn_rect.collidepoint(mx, my):
                        # Parse hex fields → build uint64 o, p, q → switch to stage 2
                        try:
                            p_re_bits = int(state.debug_p_re_hex or "0", 16)
                            p_im_bits = int(state.debug_p_im_hex or "0", 16)
                            if p_re_bits < 0 or p_re_bits > 0xFFFFFFFF:
                                raise ValueError("p Re out of range")
                            if p_im_bits < 0 or p_im_bits > 0xFFFFFFFF:
                                raise ValueError("p Im out of range")
                            p = (p_im_bits << 32) | p_re_bits
                            q_re_bits = int(state.debug_q_re_hex or "0", 16)
                            q_im_bits = int(state.debug_q_im_hex or "0", 16)
                            if q_re_bits < 0 or q_re_bits > 0xFFFFFFFF:
                                raise ValueError("q Re out of range")
                            if q_im_bits < 0 or q_im_bits > 0xFFFFFFFF:
                                raise ValueError("q Im out of range")
                            q = (q_im_bits << 32) | q_re_bits
                            o_re_bits = int(state.debug_o_re_hex or "0", 16)
                            o_im_bits = int(state.debug_o_im_hex or "0", 16)
                            if o_re_bits < 0 or o_re_bits > 0xFFFFFFFF:
                                raise ValueError("o Re out of range")
                            if o_im_bits < 0 or o_im_bits > 0xFFFFFFFF:
                                raise ValueError("o Im out of range")
                            o = (o_im_bits << 32) | o_re_bits
                            o_re, o_im = decode_o_display(o)
                            p_re, p_im = decode_p_display(p)
                            q_re, q_im = decode_q_display(q)
                            state.stage2_o = o
                            state.stage2_o_re = o_re
                            state.stage2_o_im = o_im
                            state.stage2_p = p
                            state.stage2_p_re = p_re
                            state.stage2_p_im = p_im
                            state.stage2_q = q
                            state.stage2_q_re = q_re
                            state.stage2_q_im = q_im
                            cache_clear_stage2()
                            state.stage = 2
                            state.needs_redraw = True
                            state.status_msg = f"Debug → Stage 2  o=0x{o:016X}  p=0x{p:016X}  q=0x{q:016X}"
                            state.status_color = CLR_STAGE_ACT
                        except ValueError as e:
                            state.status_msg = f"Invalid hex: {e}"
                            state.status_color = CLR_ERROR
                    elif hasattr(state, '_copy_mnemonic_btn_rect') and state._copy_mnemonic_btn_rect.collidepoint(mx, my):
                        if state.decoded_mnemonic:
                            try:
                                copy_to_clipboard(state.decoded_mnemonic)
                                state.status_msg = "Mnemonic copied to clipboard"
                                state.status_color = CLR_SUCCESS
                            except Exception as e:
                                state.status_msg = f"Clipboard error: {e}"
                                state.status_color = CLR_ERROR
                    elif hasattr(state, '_salt_rect') and state._salt_rect.collidepoint(mx, my):
                        state.salt_focused = True
                        state.input_focused = False
                        state.argon2_iter_focused = False
                        state.maxiter_focused = False
                        state.debug_o_re_focused = False
                        state.debug_o_im_focused = False
                        state.debug_p_re_focused = False
                        state.debug_p_im_focused = False
                        state.debug_q_re_focused = False
                        state.debug_q_im_focused = False
                    elif hasattr(state, '_sha512_btn_rect') and state._sha512_btn_rect.collidepoint(mx, my):
                        if state.decoded_mnemonic:
                            try:
                                data = (state.decoded_mnemonic + state.salt_text).encode("utf-8")
                                digest = hashlib.sha512(data).hexdigest()
                                copy_to_clipboard(digest)
                                state.status_msg = f"SHA512 copied: {digest[:32]}..."
                                state.status_color = CLR_SUCCESS
                            except Exception as e:
                                state.status_msg = f"SHA512 error: {e}"
                                state.status_color = CLR_ERROR
                        else:
                            state.status_msg = "No mnemonic to hash"
                            state.status_color = CLR_WARNING
                    elif hasattr(state, '_scheme_rects'):
                        for i, r in enumerate(state._scheme_rects):
                            if r.collidepoint(mx, my):
                                state.palette_idx = i
                                state.needs_redraw = True
                                break
                    if hasattr(state, '_maxiter_rect') and state._maxiter_rect.collidepoint(mx, my):
                        state.maxiter_focused = True
                        state.input_focused = False
                        state.argon2_iter_focused = False
                        state.debug_p_re_focused = False
                        state.debug_p_im_focused = False
                    continue

                # Fractal area clicks
                if event.button == 1:  # Left click
                    # Stretch correction point selection
                    if state.stretch_mode:
                        re, im = state.screen_to_complex(mx, my)
                        if state.stretch_p1 is None:
                            state.stretch_p1 = (re, im)
                            state.status_msg = f"Stretch P1=({re:.6f},{im:.6f}) — click P2"
                            state.status_color = CLR_INFO
                        else:
                            p1_re, p1_im = state.stretch_p1
                            dist = math.hypot(re - p1_re, im - p1_im)
                            theta = math.atan2(im - p1_im, re - p1_re)
                            # Reference length C: fraction of viewport width in complex units
                            c_ref = STRETCH_C_FRACTION * state.pixel_step() * state.vp_w
                            if dist < 1e-15:
                                state.status_msg = "P1 and P2 are the same point — ignored"
                                state.status_color = CLR_ERROR
                            else:
                                scale = c_ref / dist
                                state.stretch_corrections.append((p1_re, p1_im, theta, scale))
                                state.needs_repalette = True
                                state.status_msg = (
                                    f"Stretch correction applied: θ={math.degrees(theta):.1f}° "
                                    f"scale={scale:.4f} (press X again to compound, Z to clear)"
                                )
                                state.status_color = CLR_ADVANCE
                            state.stretch_mode = False
                            state.stretch_p1 = None
                        continue

                    # Check if clicking near an encoded point marker
                    if state.encoded_points and not state.select_mode:
                        best_dist = POINT_CLICK_THRESHOLD_PX
                        best_idx = None
                        for i, (re, im, _, _) in enumerate(state.encoded_points):
                            sx, sy = state.complex_to_screen(re, im)
                            dist = math.hypot(mx - sx, my - sy)
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = i
                        if best_idx is not None:
                            state.selected_point_idx = best_idx
                            state.selected_decoded_idx = None
                            p_label = best_idx + 1 + (0 if state.stage == 1 else state.points_per_stage)
                            state.status_msg = f"Selected P{p_label} — N/P to cycle"
                            state.status_color = MARKER_COLORS[best_idx % len(MARKER_COLORS)]
                            continue

                    # Check if clicking near a selected (decoded) point marker
                    if state.selected_points and not state.select_mode:
                        best_dist = POINT_CLICK_THRESHOLD_PX
                        best_idx = None
                        for i, (re_raw, im_raw) in enumerate(state.selected_points):
                            sx, sy = state.complex_to_screen(fixed_to_f64(re_raw), fixed_to_f64(im_raw))
                            dist = math.hypot(mx - sx, my - sy)
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = i
                        if best_idx is not None:
                            state.selected_decoded_idx = best_idx
                            state.selected_point_idx = None
                            state.status_msg = f"Selected S{best_idx + 1} — N/P to cycle"
                            state.status_color = MARKER_COLORS[best_idx % len(MARKER_COLORS)]
                            continue

                    if state.select_mode and state.stage == 1 and len(state.selected_points) < state.points_per_stage:
                        # --- Stage-1 select-points flow ---
                        re_raw, im_raw = state.screen_to_complex_fixed(mx, my)
                        re = fixed_to_f64(re_raw)
                        im = fixed_to_f64(im_raw)

                        try:
                            _bits, leaf_rect, valid, _sel_path = decode_full(
                                re_raw, im_raw, BITS_PER_POINT,
                                area=ENCODE_AREA, params=GUI_PARAMS,
                                o=STAGE1_O, p=STAGE1_P, q=STAGE1_Q,
                                path_prefix="O",
                            )
                        except Exception as e:
                            state.status_msg = f"Decode error: {e}"
                            state.status_color = CLR_ERROR
                            valid = False

                        if not valid:
                            state.highlighted_leaf_rect = None
                            state.needs_repalette = True
                            if state.status_color != CLR_ERROR:
                                state.status_msg = f"Bad point at ({re:.6f}, {im:.6f}) — excluded by contraction"
                                state.status_color = CLR_ERROR
                        else:
                            state.selected_points.append((re_raw, im_raw))
                            idx = len(state.selected_points)
                            state.status_msg = f"Point {idx}/{state.points_per_stage} selected at ({re:.6f}, {im:.6f})"
                            state.status_color = CLR_WARNING
                            state.highlighted_leaf_rect = leaf_rect
                            state.needs_repalette = True

                        if len(state.selected_points) == state.points_per_stage:
                            try:
                                stage1_bits, s1_steps, s1_rects = decode_points(
                                    state.selected_points, p=STAGE1_P)
                                state.decoded_stage1_bits = stage1_bits
                                state.argon2_stage1_bits = stage1_bits
                                state.argon2_digest = ""
                                stage1_hex = bits_to_hex(stage1_bits)
                                state.decoded_mnemonic = f"[stage 1: {stage1_hex}]"
                                state.selected_steps = s1_steps
                                state.selected_final_rects = s1_rects
                                state.status_msg = f"Decoded {state.points_per_stage} pts → {state.bits_per_stage} bits (stage 1). Run Argon2 → stage 2."
                                state.status_color = CLR_SUCCESS
                                state.select_mode = False
                            except Exception as e:
                                state.status_msg = f"Decode error: {e}"
                                state.status_color = CLR_ERROR
                                state.select_mode = False

                    elif state.select_mode and state.stage == 2 and len(state.stage2_selected_points) < state.points_per_stage:
                        # --- Stage-2 select-points flow ---
                        re_raw, im_raw = state.screen_to_complex_fixed(mx, my)
                        re = fixed_to_f64(re_raw)
                        im = fixed_to_f64(im_raw)

                        try:
                            _bits, leaf_rect, valid, _sel_path = decode_full(
                                re_raw, im_raw, BITS_PER_POINT,
                                area=ENCODE_AREA, params=GUI_PARAMS,
                                o=state.stage2_o, p=state.stage2_p, q=state.stage2_q,
                                path_prefix="O",
                            )
                        except Exception as e:
                            state.status_msg = f"Decode error: {e}"
                            state.status_color = CLR_ERROR
                            valid = False

                        if not valid:
                            state.highlighted_leaf_rect = None
                            state.needs_repalette = True
                            if state.status_color != CLR_ERROR:
                                state.status_msg = f"Bad point at ({re:.6f}, {im:.6f}) — excluded by contraction"
                                state.status_color = CLR_ERROR
                        else:
                            state.stage2_selected_points.append((re_raw, im_raw))
                            idx = len(state.stage2_selected_points)
                            state.status_msg = f"Point {idx}/{state.points_per_stage} selected at ({re:.6f}, {im:.6f}) (stage 2)"
                            state.status_color = CLR_WARNING
                            state.highlighted_leaf_rect = leaf_rect
                            state.needs_repalette = True

                        if len(state.stage2_selected_points) == state.points_per_stage:
                            try:
                                stage2_bits, s2_steps, s2_rects = decode_points(
                                    state.stage2_selected_points, o=state.stage2_o, p=state.stage2_p, q=state.stage2_q)
                                state.decoded_stage2_bits = stage2_bits
                                stage2_hex = bits_to_hex(stage2_bits)
                                state.stage2_selected_steps = s2_steps
                                state.stage2_selected_final_rects = s2_rects
                                # Combine stage1 + stage2 → 128 entropy bits → BIP39
                                if state.decoded_stage1_bits is not None:
                                    all_entropy = state.decoded_stage1_bits + stage2_bits
                                    mnemonic = bits_to_mnemonic(all_entropy)
                                    state.decoded_mnemonic = mnemonic
                                    state.status_msg = f"Decoded all {state.points_per_stage * 2} pts → {state.entropy_bits} bits → BIP39 mnemonic"
                                    state.status_color = CLR_SUCCESS
                                else:
                                    state.decoded_mnemonic = f"[stage 2: {stage2_hex}] (stage 1 missing)"
                                    state.status_msg = f"Decoded {state.points_per_stage} pts → {state.bits_per_stage} bits (stage 2)"
                                    state.status_color = CLR_SUCCESS
                                state.select_mode = False
                            except Exception as e:
                                state.status_msg = f"Decode error: {e}"
                                state.status_color = CLR_ERROR
                                state.select_mode = False
                    else:
                        dragging = True
                        drag_start = (mx, my)

                elif event.button == 4:  # Scroll up = zoom in
                    # Zoom toward cursor (one sqrt(2) step)
                    re, im = state.screen_to_complex(mx, my)
                    state.zoom_exp += 1
                    step = state.pixel_step()
                    state.center_re = re - (mx - state.vp_w / 2) * step
                    state.center_im = im - (my - state.vp_h / 2) * step
                    state.needs_redraw = True

                elif event.button == 5:  # Scroll down = zoom out
                    re, im = state.screen_to_complex(mx, my)
                    state.zoom_exp -= 1
                    step = state.pixel_step()
                    state.center_re = re - (mx - state.vp_w / 2) * step
                    state.center_im = im - (my - state.vp_h / 2) * step
                    state.needs_redraw = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
                    drag_start = None

            elif event.type == pygame.MOUSEMOTION:
                if dragging and drag_start:
                    dx = event.pos[0] - drag_start[0]
                    dy = event.pos[1] - drag_start[1]
                    step = state.pixel_step()
                    state.center_re -= dx * step
                    state.center_im -= dy * step
                    drag_start = event.pos
                    state.needs_redraw = True

        # Render fractal if needed (progressive: coarse → fine)
        if state.needs_redraw:
            state.progressive_block = PROGRESSIVE_INITIAL_BLOCK
            state.needs_redraw = False

        if state.progressive_block is not None:
            render_fractal(state, block_size=state.progressive_block)
            state.needs_repalette = True
            if state.progressive_block <= 1:
                state.progressive_block = None  # done
            else:
                state.progressive_block //= 2  # refine next frame

        if state.needs_repalette:
            fractal_surface = apply_palette(
                state.pixel_buf, PALETTE_NAMES[state.palette_idx],
                state.vp_w, state.vp_h, state,
            )
            state.needs_repalette = False
            state._stretched_surface = None  # invalidate stretch cache

        # Draw
        if fractal_surface:
            if state.stretch_corrections:
                # Compute stretched surface once, cache until palette/view changes
                if state._stretched_surface is None and state.progressive_block is None:
                    state._stretched_surface = apply_stretch_corrections(
                        fractal_surface, state)
                display_surface = state._stretched_surface or fractal_surface
            else:
                display_surface = fractal_surface
            screen.blit(display_surface, (0, 0))

        # Draw bisection area rectangles (V key or selected point)
        if state.show_areas or state.selected_point_idx is not None or state.selected_decoded_idx is not None:
            active = _get_active_steps(state)
            step_offset = 0
            for steps_data, frect, pt_color in active:
                # Map global focus_step to per-point local step index
                local_focus = None
                if state.area_focus_step is not None and state.show_areas:
                    local_idx = state.area_focus_step - step_offset
                    if 0 <= local_idx < len(steps_data):
                        local_focus = local_idx
                    elif local_idx < 0 or local_idx >= len(steps_data):
                        step_offset += len(steps_data)
                        continue
                step_offset += len(steps_data)
                draw_bisect_rects(screen, state, steps_data, frect, pt_color,
                                  focus_step=local_focus)

        # Draw encoded points (stage-aware: P1,P2 for stage 1; P3,P4 for stage 2)
        point_offset = 0 if state.stage == 1 else state.points_per_stage
        for i, (re, im, re_raw, im_raw) in enumerate(state.encoded_points):
            sx, sy = state.complex_to_screen(re, im)
            if 0 <= sx < state.vp_w and 0 <= sy < state.vp_h:
                color = MARKER_COLORS[i % len(MARKER_COLORS)]
                selected = (i == state.selected_point_idx)
                label = f"P{i + 1 + point_offset}" + (" *" if selected else "")
                draw_marker(screen, sx, sy, color, label, small_font)

        # Draw selected points (stage 1 and stage 2)
        for i, (re_raw, im_raw) in enumerate(state.selected_points):
            re = fixed_to_f64(re_raw)
            im = fixed_to_f64(im_raw)
            sx, sy = state.complex_to_screen(re, im)
            if 0 <= sx < state.vp_w and 0 <= sy < state.vp_h:
                color = MARKER_COLORS[i % len(MARKER_COLORS)]
                selected = (i == state.selected_decoded_idx)
                label = f"S{i+1}" + (" *" if selected else "")
                draw_marker(screen, sx, sy, color, label, small_font)
        for i, (re_raw, im_raw) in enumerate(state.stage2_selected_points):
            re = fixed_to_f64(re_raw)
            im = fixed_to_f64(im_raw)
            sx, sy = state.complex_to_screen(re, im)
            if 0 <= sx < state.vp_w and 0 <= sy < state.vp_h:
                color = MARKER_COLORS[(i + state.points_per_stage) % len(MARKER_COLORS)]
                label = f"S{i + 1 + state.points_per_stage}"
                draw_marker(screen, sx, sy, color, label, small_font)

        # Draw panel
        draw_panel(screen, state, font, small_font)

        # Coordinate display at cursor (debug mode only)
        if state.debug_mode:
            mx, my = pygame.mouse.get_pos()
            if my < state.vp_h:
                re, im = state.screen_to_complex(mx, my)
                coord_txt = small_font.render(f"({re:.8f}, {im:.8f})", True, (200, 200, 200))
                screen.blit(coord_txt, (state.vp_w - coord_txt.get_width() - 5, 5))

        pygame.display.flip()
        clock.tick(30)

    if args.cache_size > 0:
        cache_destroy()
        cache_destroy_stage2()
    pygame.quit()


if __name__ == "__main__":
    main()
