"""
Python FFI bridge to the Rust Burning Ship fractal engine.

Provides encode/decode functions that map bit arrays to fractal locations.
All determinism-critical values are passed as integers (raw Fixed i64),
never as floating-point.
"""

import ctypes
import os
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Argon2 I/O sizes (bytes)
ARGON2_INPUT_BYTES = 8            # 64-bit stage-1 entropy
ARGON2_DIGEST_BYTES = 32          # 256-bit digest output

# Argon2 profiles (must match Rust PROFILE_BASIC / PROFILE_ADVANCED / PROFILE_GREAT_WALL)
PROFILE_BASIC      = 0  # mobile-accessible:   1 GiB, p=1, t=2  (Argon2d)
PROFILE_ADVANCED   = 1  # desktop-only:       32 GiB, p=1, t=2  (Argon2d)
PROFILE_GREAT_WALL = 2  # server-class:      128 GiB, p=1, t=2  (Argon2d)

# ---------------------------------------------------------------------------
# Load the Rust shared library
# ---------------------------------------------------------------------------

_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_dir, "rust_engine", "target", "release", "libburning_ship_engine.so")

if not os.path.exists(_lib_path):
    print(f"ERROR: {_lib_path} not found. Run 'cargo build --release' in rust_engine/ first.",
          file=sys.stderr)
    sys.exit(1)

_lib = ctypes.CDLL(_lib_path)

# ---------------------------------------------------------------------------
# FFI signatures
# ---------------------------------------------------------------------------

_lib.bs_escape_count.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_uint32]
_lib.bs_escape_count.restype = ctypes.c_int32

_lib.bs_render_viewport.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint8),
]
_lib.bs_render_viewport.restype = None

_lib.bs_render_viewport_generic.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32,
    ctypes.c_uint64,                  # o (orbit seed)
    ctypes.c_uint64,                  # p (additive perturbation)
    ctypes.c_uint64,                  # q (linear perturbation)
    ctypes.POINTER(ctypes.c_uint8),
]
_lib.bs_render_viewport_generic.restype = None

_lib.bs_encode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # area (Fixed)
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,  # max_iter, target_good, max_flood_points
    ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32,  # min_grid_cells, p_max_shift, exclusion_threshold_num
    ctypes.c_uint64,  # rng_seed
    ctypes.c_uint64,  # o (orbit seed)
    ctypes.c_uint64,  # p (additive perturbation)
    ctypes.c_uint64,  # q (linear perturbation)
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,  # path_prefix, path_prefix_len
]
_lib.bs_encode.restype = ctypes.c_void_p

_lib.bs_encode_result_point.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)]
_lib.bs_encode_result_point.restype = None

_lib.bs_encode_result_bits.argtypes = [ctypes.c_void_p]
_lib.bs_encode_result_bits.restype = ctypes.c_uint32

_lib.bs_encode_result_num_steps.argtypes = [ctypes.c_void_p]
_lib.bs_encode_result_num_steps.restype = ctypes.c_uint32

_lib.bs_encode_result_path.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32]
_lib.bs_encode_result_path.restype = ctypes.c_uint32

_lib.bs_encode_result_step.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32),  # split_coord (raw Fixed), split_vertical
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32),  # bit, chose_larger
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),  # contraction_num, contraction_den
    ctypes.POINTER(ctypes.c_uint32),  # islands_found
]
_lib.bs_encode_result_step.restype = None

_lib.bs_encode_result_step_num_islands.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_lib.bs_encode_result_step_num_islands.restype = ctypes.c_uint32

_lib.bs_encode_result_step_island.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_uint64),
]
_lib.bs_encode_result_step_island.restype = None

_lib.bs_encode_result_step_area.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
]
_lib.bs_encode_result_step_area.restype = None

_lib.bs_encode_result_final_rect.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
]
_lib.bs_encode_result_final_rect.restype = None

_lib.bs_encode_result_free.argtypes = [ctypes.c_void_p]
_lib.bs_encode_result_free.restype = None

_lib.bs_encode_result_num_seeds.argtypes = [ctypes.c_void_p]
_lib.bs_encode_result_num_seeds.restype = ctypes.c_uint32

_lib.bs_encode_result_seeds.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
]
_lib.bs_encode_result_seeds.restype = None

_lib.bs_encode_with_seeds.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # area (Fixed)
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,
    ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,  # o
    ctypes.c_uint64,  # p
    ctypes.c_uint64,  # q
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,  # path_prefix
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,  # seeds_buf, num_seeds
]
_lib.bs_encode_with_seeds.restype = ctypes.c_void_p

_lib.bs_decode.argtypes = [
    ctypes.c_int64, ctypes.c_int64, ctypes.c_uint32,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # area (Fixed)
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,  # max_iter, target_good, max_flood_points
    ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32,  # min_grid_cells, p_max_shift, exclusion_threshold_num
    ctypes.c_uint64,  # rng_seed
    ctypes.c_uint64,  # o (orbit seed)
    ctypes.c_uint64,  # p (additive perturbation)
    ctypes.c_uint64,  # q (linear perturbation)
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,  # path_prefix, path_prefix_len
    ctypes.POINTER(ctypes.c_uint8),  # out_bits
]
_lib.bs_decode.restype = None

_lib.bs_decode_full.argtypes = [
    ctypes.c_int64, ctypes.c_int64, ctypes.c_uint32,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # area (Fixed)
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,
    ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,  # o (orbit seed)
    ctypes.c_uint64,  # p (additive perturbation)
    ctypes.c_uint64,  # q (linear perturbation)
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,  # path_prefix, path_prefix_len
    ctypes.POINTER(ctypes.c_uint8),  # out_bits
    ctypes.POINTER(ctypes.c_int64),  # out_rect (raw Fixed)
    ctypes.POINTER(ctypes.c_uint8),  # out_valid
    ctypes.POINTER(ctypes.c_uint8),  # out_path buf
    ctypes.c_uint32,                 # out_path_buf_len
    ctypes.POINTER(ctypes.c_uint32), # out_path_len
]
_lib.bs_decode_full.restype = None

_lib.bs_get_precision.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
_lib.bs_get_precision.restype = None

# Render cache management
_lib.bs_cache_init.argtypes = [ctypes.c_uint64]
_lib.bs_cache_init.restype = None

_lib.bs_cache_clear.argtypes = []
_lib.bs_cache_clear.restype = None

_lib.bs_cache_destroy.argtypes = []
_lib.bs_cache_destroy.restype = None

_lib.bs_cache_len.argtypes = []
_lib.bs_cache_len.restype = ctypes.c_uint64

# Stage-2 render cache management
_lib.bs_cache_init_stage2.argtypes = [ctypes.c_uint64]
_lib.bs_cache_init_stage2.restype = None

_lib.bs_cache_clear_stage2.argtypes = []
_lib.bs_cache_clear_stage2.restype = None

_lib.bs_cache_destroy_stage2.argtypes = []
_lib.bs_cache_destroy_stage2.restype = None

# Argon2 iterative hashing
_lib.bs_argon2_hash.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # input_ptr (8 bytes)
    ctypes.c_uint32,                  # gui_iterations
    ctypes.c_uint8,                   # profile (0=Standard, 1=Fortress)
    ctypes.POINTER(ctypes.c_uint8),  # out_digest (32 bytes)
]
_lib.bs_argon2_hash.restype = None

# Argon2 single-pass (for progress-aware iteration from Python)
_lib.bs_argon2_single.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # input_ptr (variable length)
    ctypes.c_uint32,                  # input_len
    ctypes.c_uint8,                   # profile (0=Standard, 1=Fortress)
    ctypes.POINTER(ctypes.c_uint8),  # out_digest (32 bytes)
]
_lib.bs_argon2_single.restype = None



# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------

class DiscoveryParams:
    """Parameters for island discovery and bisection.

    All parameters are integers — no floating-point.

    min_grid_cells: p0 is the largest power-of-2 pixel size such that
                    (width * height) / p0² >= min_grid_cells.
    p_max_shift: maximum flood-fill pixel size = area_width >> p_max_shift
    exclusion_threshold_num: stop when excluded_cells/total_cells >= threshold/256
    """

    def __init__(self, max_iter=1024, target_good=100, max_flood_points=50000,
                 min_grid_cells=4096, p_max_shift=3, exclusion_threshold_num=230,
                 rng_seed=0x42):
        self.max_iter = max_iter
        self.target_good = target_good
        self.max_flood_points = max_flood_points
        self.min_grid_cells = min_grid_cells
        self.p_max_shift = p_max_shift
        self.exclusion_threshold_num = exclusion_threshold_num
        self.rng_seed = rng_seed


def _query_frac_bits():
    """Query FRAC_BITS from the Rust engine at load time."""
    fb = ctypes.c_uint32()
    ib = ctypes.c_uint32()
    _lib.bs_get_precision(ctypes.byref(fb), ctypes.byref(ib))
    return fb.value

# Fixed-point scale factor (set once at module load).
FRAC_BITS = _query_frac_bits()
FIXED_ONE = 1 << FRAC_BITS


def fixed_from_f64(v):
    """Convert a Python float to a raw Fixed i64 value."""
    return int(v * FIXED_ONE)


def fixed_to_f64(raw):
    """Convert a raw Fixed i64 value to a Python float (for display only)."""
    return raw / FIXED_ONE


class Rect:
    """Rectangular area in the complex plane (raw Fixed i64 values)."""

    def __init__(self, re_min, re_max, im_min, im_max):
        """Create from raw Fixed i64 values."""
        self.re_min = re_min
        self.re_max = re_max
        self.im_min = im_min
        self.im_max = im_max

    @classmethod
    def from_f64(cls, re_min, re_max, im_min, im_max):
        """Create from f64 values (convenience; rounds to nearest Fixed)."""
        return cls(fixed_from_f64(re_min), fixed_from_f64(re_max),
                   fixed_from_f64(im_min), fixed_from_f64(im_max))

    def re_min_f64(self):
        return fixed_to_f64(self.re_min)

    def re_max_f64(self):
        return fixed_to_f64(self.re_max)

    def im_min_f64(self):
        return fixed_to_f64(self.im_min)

    def im_max_f64(self):
        return fixed_to_f64(self.im_max)

    def __repr__(self):
        return (f"Rect(re=[{self.re_min_f64():.6f}, {self.re_max_f64():.6f}], "
                f"im=[{self.im_min_f64():.6f}, {self.im_max_f64():.6f}])")


# Default area: canonical Burning Ship fractal region [-2.5, 1.5] x [-2.0, 1.5].
# Stored as raw Fixed i64 to avoid f64 rounding.
DEFAULT_AREA = Rect(
    fixed_from_f64(-2.5), fixed_from_f64(1.5),
    fixed_from_f64(-2.0), fixed_from_f64(1.5),
)


class EncodeResult:
    """Result of encoding a bit array."""

    def __init__(self, handle):
        self._handle = handle
        if not handle:
            raise RuntimeError("bs_encode returned NULL handle")

        # Extract point as raw i64 (canonical representation)
        re_raw = ctypes.c_int64()
        im_raw = ctypes.c_int64()
        _lib.bs_encode_result_point(handle, ctypes.byref(re_raw), ctypes.byref(im_raw))
        self.point_re_raw = re_raw.value
        self.point_im_raw = im_raw.value
        # f64 views for display convenience
        self.point_re = fixed_to_f64(self.point_re_raw)
        self.point_im = fixed_to_f64(self.point_im_raw)

        self.bits_consumed = _lib.bs_encode_result_bits(handle)
        self.num_steps = _lib.bs_encode_result_num_steps(handle)

        # Extract final rect as raw Fixed i64
        rmin = ctypes.c_int64()
        rmax = ctypes.c_int64()
        imin = ctypes.c_int64()
        imax = ctypes.c_int64()
        _lib.bs_encode_result_final_rect(handle,
                                          ctypes.byref(rmin), ctypes.byref(rmax),
                                          ctypes.byref(imin), ctypes.byref(imax))
        self.final_rect = Rect(rmin.value, rmax.value, imin.value, imax.value)

        # Extract path string
        path_len = _lib.bs_encode_result_path(handle, None, 0)
        buf = (ctypes.c_uint8 * (path_len + 1))()
        _lib.bs_encode_result_path(handle, buf, path_len + 1)
        self.path = bytes(buf[:path_len]).decode("ascii")

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.bs_encode_result_free(self._handle)
            self._handle = None

    def _check_handle(self):
        if not self._handle:
            raise RuntimeError("EncodeResult handle already freed")

    def get_step(self, idx):
        """Get info about bisection step `idx`.

        split_coord is a raw Fixed i64 value.
        contraction_num / contraction_den give the exact rational contraction.
        contraction is the f64 approximation for display.
        """
        self._check_handle()
        sc = ctypes.c_int64()
        sv = ctypes.c_int32()
        bit = ctypes.c_uint8()
        cl = ctypes.c_int32()
        cn = ctypes.c_uint64()
        cd = ctypes.c_uint64()
        isl = ctypes.c_uint32()
        _lib.bs_encode_result_step(self._handle, idx,
                                    ctypes.byref(sc), ctypes.byref(sv),
                                    ctypes.byref(bit), ctypes.byref(cl),
                                    ctypes.byref(cn), ctypes.byref(cd),
                                    ctypes.byref(isl))
        den = cd.value if cd.value != 0 else 1
        return {
            'split_coord': sc.value,
            'split_coord_f64': fixed_to_f64(sc.value),
            'split_vertical': bool(sv.value),
            'bit': bit.value,
            'chose_larger': bool(cl.value),
            'contraction_num': cn.value,
            'contraction_den': den,
            'contraction': cn.value / den,
            'islands_found': isl.value,
        }

    def get_step_area(self, idx):
        """Get the area rectangle for bisection step `idx` (raw Fixed i64)."""
        self._check_handle()
        rmin = ctypes.c_int64()
        rmax = ctypes.c_int64()
        imin = ctypes.c_int64()
        imax = ctypes.c_int64()
        _lib.bs_encode_result_step_area(self._handle, idx,
                                         ctypes.byref(rmin), ctypes.byref(rmax),
                                         ctypes.byref(imin), ctypes.byref(imax))
        return Rect(rmin.value, rmax.value, imin.value, imax.value)

    def get_step_islands(self, idx):
        """Get island barycenters and pixel counts for step `idx`."""
        self._check_handle()
        n = _lib.bs_encode_result_step_num_islands(self._handle, idx)
        islands = []
        for j in range(n):
            re = ctypes.c_int64()
            im = ctypes.c_int64()
            px = ctypes.c_uint64()
            _lib.bs_encode_result_step_island(self._handle, idx, j,
                                               ctypes.byref(re), ctypes.byref(im),
                                               ctypes.byref(px))
            islands.append({
                'center_re': fixed_to_f64(re.value),
                'center_im': fixed_to_f64(im.value),
                'pixel_count': px.value,
            })
        return islands

    def get_all_steps(self):
        """Get all step data including areas and islands, for visualization."""
        steps = []
        for i in range(self.num_steps):
            step = self.get_step(i)
            step['area'] = self.get_step_area(i)
            step['islands'] = self.get_step_islands(i)
            steps.append(step)
        return steps

    def get_seeds_blob(self):
        """Get inherited seeds as an opaque bytes blob for passing to encode_with_seeds."""
        self._check_handle()
        n = _lib.bs_encode_result_num_seeds(self._handle)
        if n == 0:
            return b"", 0
        buf_len = n * 24
        buf = (ctypes.c_uint8 * buf_len)()
        _lib.bs_encode_result_seeds(self._handle, buf, buf_len)
        return bytes(buf), n


def encode_with_seeds(bits, seeds_blob, num_seeds, area=None, params=None,
                      o=0, p=0, q=0, path_prefix="O"):
    """Encode with initial inherited seeds (for incremental encoding).

    seeds_blob/num_seeds come from a previous EncodeResult.get_seeds_blob().
    """
    if area is None:
        area = DEFAULT_AREA
    if params is None:
        params = DiscoveryParams()

    bits_arr = (ctypes.c_uint8 * len(bits))(*bits)
    pp_arr, pp_len = _path_prefix_args(path_prefix)

    if num_seeds > 0 and seeds_blob:
        seeds_arr = (ctypes.c_uint8 * len(seeds_blob))(*seeds_blob)
    else:
        seeds_arr = None
        num_seeds = 0

    handle = _lib.bs_encode_with_seeds(
        bits_arr, len(bits),
        area.re_min, area.re_max, area.im_min, area.im_max,
        params.max_iter, params.target_good, params.max_flood_points,
        params.min_grid_cells, params.p_max_shift, params.exclusion_threshold_num,
        params.rng_seed,
        o, p, q,
        pp_arr, pp_len,
        seeds_arr, num_seeds,
    )

    return EncodeResult(handle)


def get_precision():
    """Return (frac_bits, int_bits) of the Rust fixed-point type."""
    fb = ctypes.c_uint32()
    ib = ctypes.c_uint32()
    _lib.bs_get_precision(ctypes.byref(fb), ctypes.byref(ib))
    return fb.value, ib.value


def get_engine_version():
    """Return the engine algorithm version string (e.g. '0.1.0')."""
    buf = (ctypes.c_uint8 * 32)()
    _lib.bs_engine_version(buf, 32)
    return bytes(buf).split(b'\x00', 1)[0].decode("ascii")


def escape_count(c_re, c_im, max_iter):
    """Compute BS escape count for a single point. Returns int or None."""
    r = _lib.bs_escape_count(c_re, c_im, max_iter)
    return None if r < 0 else r


def _path_prefix_args(path_prefix):
    """Convert a path_prefix string to (ctypes pointer, length) pair."""
    if path_prefix:
        buf = path_prefix.encode("ascii")
        arr = (ctypes.c_uint8 * len(buf))(*buf)
        return arr, len(buf)
    return None, 0


def encode(bits, area=None, params=None, o=0, p=0, q=0, path_prefix="O"):
    """
    Encode a bit array into a fractal location.

    Args:
        bits: list/bytes of 0s and 1s
        area: Rect for the initial area (default: DEFAULT_AREA)
        params: DiscoveryParams (default: sensible defaults)
        o: uint64 orbit seed parameter
        p: uint64 additive perturbation parameter (0 = canonical formula)
        q: uint64 linear perturbation parameter (0 = no linear term)
        path_prefix: accumulated path from prior points (default: "O")

    Returns:
        EncodeResult with point_re, point_im, path, etc.
    """
    if area is None:
        area = DEFAULT_AREA
    if params is None:
        params = DiscoveryParams()

    bits_arr = (ctypes.c_uint8 * len(bits))(*bits)
    pp_arr, pp_len = _path_prefix_args(path_prefix)

    handle = _lib.bs_encode(
        bits_arr, len(bits),
        area.re_min, area.re_max, area.im_min, area.im_max,
        params.max_iter, params.target_good, params.max_flood_points,
        params.min_grid_cells, params.p_max_shift, params.exclusion_threshold_num,
        params.rng_seed,
        o,
        p,
        q,
        pp_arr, pp_len,
    )

    return EncodeResult(handle)


def decode(point_re_raw, point_im_raw, num_bits, area=None, params=None, o=0, p=0, q=0,
           path_prefix="O"):
    """
    Decode a fractal location back to a bit array.

    Args:
        point_re_raw, point_im_raw: the encoded point as raw i64 Fixed values
        num_bits: number of bits to decode
        area: Rect for the initial area (must match encode)
        params: DiscoveryParams (must match encode)
        o: uint64 orbit seed parameter
        p: uint64 additive perturbation parameter (0 = canonical formula)
        q: uint64 linear perturbation parameter (0 = no linear term)
        path_prefix: must match the prefix used during encoding

    Returns:
        list of 0s and 1s
    """
    if area is None:
        area = DEFAULT_AREA
    if params is None:
        params = DiscoveryParams()

    pp_arr, pp_len = _path_prefix_args(path_prefix)
    out = (ctypes.c_uint8 * num_bits)()

    _lib.bs_decode(
        point_re_raw, point_im_raw, num_bits,
        area.re_min, area.re_max, area.im_min, area.im_max,
        params.max_iter, params.target_good, params.max_flood_points,
        params.min_grid_cells, params.p_max_shift, params.exclusion_threshold_num,
        params.rng_seed,
        o,
        p,
        q,
        pp_arr, pp_len,
        out,
    )

    return list(out)


def decode_full(point_re_raw, point_im_raw, num_bits, area=None, params=None, o=0, p=0, q=0,
                path_prefix="O"):
    """
    Decode a fractal location back to a bit array, also returning the leaf
    rectangle and a validity flag.

    Returns:
        (bits, leaf_rect, valid, path) where bits is a list of 0/1,
        leaf_rect is a Rect, valid is bool, and path is the accumulated
        path string after this decode.
    """
    if area is None:
        area = DEFAULT_AREA
    if params is None:
        params = DiscoveryParams()

    # Path buffer: prefix + num_bits direction chars + NUL
    path_buf_len = len(path_prefix) + num_bits + 2
    pp_arr, pp_len = _path_prefix_args(path_prefix)
    out = (ctypes.c_uint8 * num_bits)()
    out_rect = (ctypes.c_int64 * 4)()
    out_valid = ctypes.c_uint8()
    out_path = (ctypes.c_uint8 * path_buf_len)()
    out_path_len = ctypes.c_uint32()

    _lib.bs_decode_full(
        point_re_raw, point_im_raw, num_bits,
        area.re_min, area.re_max, area.im_min, area.im_max,
        params.max_iter, params.target_good, params.max_flood_points,
        params.min_grid_cells, params.p_max_shift, params.exclusion_threshold_num,
        params.rng_seed,
        o,
        p,
        q,
        pp_arr, pp_len,
        out, out_rect, ctypes.byref(out_valid),
        out_path, path_buf_len, ctypes.byref(out_path_len),
    )

    leaf_rect = Rect(out_rect[0], out_rect[1], out_rect[2], out_rect[3])
    path = bytes(out_path[:out_path_len.value]).decode("ascii")
    return list(out), leaf_rect, bool(out_valid.value), path


def cache_init(capacity):
    """Initialize the render cache with the given number of entries."""
    _lib.bs_cache_init(capacity)


def cache_clear():
    """Clear all entries from the render cache."""
    _lib.bs_cache_clear()


def cache_destroy():
    """Destroy the render cache, freeing its memory."""
    _lib.bs_cache_destroy()


def cache_len():
    """Return the current number of entries in the render cache."""
    return _lib.bs_cache_len()


def cache_init_stage2(capacity):
    """Initialize the stage-2 render cache with the given number of entries."""
    _lib.bs_cache_init_stage2(capacity)


def cache_clear_stage2():
    """Clear all entries from the stage-2 render cache."""
    _lib.bs_cache_clear_stage2()


def cache_destroy_stage2():
    """Destroy the stage-2 render cache, freeing its memory."""
    _lib.bs_cache_destroy_stage2()


def argon2_hash(input_bytes, gui_iterations, profile=PROFILE_BASIC):
    """Run iterative Argon2d hashing on 8 bytes of stage-1 entropy.

    Args:
        input_bytes: bytes object of length 8 (64 bits of entropy)
        gui_iterations: number of hash-then-feed-back cycles (1 = single pass)
        profile: PROFILE_BASIC (0), PROFILE_ADVANCED (1), or PROFILE_GREAT_WALL (2)

    Returns:
        bytes: 32-byte (256-bit) digest
    """
    assert len(input_bytes) == ARGON2_INPUT_BYTES, f"Expected {ARGON2_INPUT_BYTES} bytes, got {len(input_bytes)}"
    inp = (ctypes.c_uint8 * ARGON2_INPUT_BYTES)(*input_bytes)
    out = (ctypes.c_uint8 * ARGON2_DIGEST_BYTES)()
    _lib.bs_argon2_hash(inp, gui_iterations, profile, out)
    return bytes(out)


def argon2_single(input_bytes, profile=PROFILE_BASIC):
    """Run a single Argon2d pass on arbitrary-length input.

    Args:
        input_bytes: bytes object (8 bytes on first call, 32 on subsequent)
        profile: PROFILE_BASIC (0), PROFILE_ADVANCED (1), or PROFILE_GREAT_WALL (2)

    Returns:
        bytes: 32-byte (256-bit) digest
    """
    n = len(input_bytes)
    inp = (ctypes.c_uint8 * n)(*input_bytes)
    out = (ctypes.c_uint8 * ARGON2_DIGEST_BYTES)()
    _lib.bs_argon2_single(inp, n, profile, out)
    return bytes(out)


def render_viewport(origin_re, origin_im, step, width, height, max_iter=2048):
    """Render an escape-count map. Returns bytes (0=non-escaping, 1-255=escape count)."""
    import numpy as np
    pixels = np.zeros(width * height, dtype=np.uint8)
    _lib.bs_render_viewport(
        origin_re, origin_im, step,
        width, height, max_iter,
        pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    return pixels.reshape(height, width)


def render_viewport_generic(origin_re, origin_im, step, width, height, p, max_iter=2048):
    """Render using perturbed Burning Ship: z <- (|Re(z)|+Re(p) + i(|Im(z)|+Im(p)))² + c."""
    import numpy as np
    pixels = np.zeros(width * height, dtype=np.uint8)
    _lib.bs_render_viewport_generic(
        origin_re, origin_im, step,
        width, height, max_iter, p,
        pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    return pixels.reshape(height, width)
