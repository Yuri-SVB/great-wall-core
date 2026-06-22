//! Canonical protocol parameters — the single source of truth for the values
//! that determine encode/decode output.
//!
//! The engine alone dictates the protocol: every caller (the Python bridge's
//! `constants.py`, great-wallet over FFI) must *read* these rather than declare
//! its own copy. A hand-mirrored copy is exactly what drifted before — the
//! wallet kept `max_iter = 64` while the protocol moved to `1024`, stalling
//! deep-zoom encodes. Exposing the values here (and over FFI via
//! `bs_encode_params` / `bs_encode_area` / `bs_bits_per_point`) makes that
//! class of drift impossible.
//!
//! Mirrors great-wall-core/burning_ship/constants.py (`GUI_PARAMS`,
//! `ENCODE_AREA`, `BITS_PER_POINT`).

use crate::discovery::DiscoveryParams;
use crate::fixed::Rect;

/// Bits encoded per fractal point — one point = one stage = one 32-bit chunk.
pub const BITS_PER_POINT: u32 = 32;

/// Escape-iteration cap for encode/decode island discovery. Intentionally
/// decoupled from (and far larger than) the render cap: as the bisection tree
/// zooms toward the Burning Ship set boundary, escape counts climb toward the
/// cap, and a low cap (64) makes almost every sample read as non-escaping,
/// starving island discovery so deep levels stall. 1024 keeps boundary-adjacent
/// points escaping. (ENGINE_VERSION 0.2.0 raised this; it is output-changing.)
pub const ENCODE_MAX_ITER: u32 = 1024;

/// Target number of "good" islands per bisection level.
pub const ENCODE_TARGET_GOOD: u32 = 32;

/// Cap on flood-fill points per island during discovery.
pub const ENCODE_MAX_FLOOD_POINTS: u64 = 256;

/// Minimum grid resolution (cells) for the discovery sampling grid.
pub const ENCODE_MIN_GRID_CELLS: u64 = 1024 * 1024;

/// Maximum perturbation shift exponent for `p`.
pub const ENCODE_P_MAX_SHIFT: u32 = 3;

/// Island-exclusion area threshold numerator (over a 1024 denominator).
pub const ENCODE_EXCLUSION_THRESHOLD_NUM: u32 = 1023;

/// Deterministic RNG seed for island discovery during encode/decode.
pub const ENCODE_RNG_SEED: u64 = 0x42;

/// The canonical encode/decode discovery parameters (constants.py `GUI_PARAMS`).
/// `max_attempts` is the engine's internal give-up limit and is not part of the
/// cross-boundary protocol surface; callers receive only the fields below plus
/// [`ENCODE_RNG_SEED`].
pub fn encode_params() -> DiscoveryParams {
    DiscoveryParams {
        max_iter: ENCODE_MAX_ITER,
        target_good: ENCODE_TARGET_GOOD,
        max_flood_points: ENCODE_MAX_FLOOD_POINTS,
        min_grid_cells: ENCODE_MIN_GRID_CELLS,
        p_max_shift: ENCODE_P_MAX_SHIFT,
        exclusion_threshold_num: ENCODE_EXCLUSION_THRESHOLD_NUM,
        max_attempts: crate::ffi::DEFAULT_MAX_ATTEMPTS,
    }
}

/// The encoding area — the Burning Ship region whose island density supports
/// 32-bit encoding (`constants.py` `ENCODE_AREA`).
pub fn encode_area() -> Rect {
    Rect::new(-2.5, 1.5, -2.0, 1.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_params_match_constants() {
        let p = encode_params();
        assert_eq!(p.max_iter, 1024);
        assert_eq!(p.target_good, 32);
        assert_eq!(p.max_flood_points, 256);
        assert_eq!(p.min_grid_cells, 1024 * 1024);
        assert_eq!(p.p_max_shift, 3);
        assert_eq!(p.exclusion_threshold_num, 1023);
        assert_eq!(BITS_PER_POINT, 32);
        assert_eq!(ENCODE_RNG_SEED, 0x42);
    }
}
