//! Landmark-path parameters: one value, [`LandmarkParams`], holds every number
//! the path depends on. Its defaults are the constants below; callers read them
//! (`bs_landmark_params`) rather than declaring their own.

use crate::protocol::CANONICAL_MAX_FLOOD_POINTS;

/// A non-negative rational `num / den`, so fractional parameters stay exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ratio {
    pub num: u32,
    pub den: u32,
}

impl Ratio {
    pub const fn new(num: u32, den: u32) -> Self {
        Ratio { num, den }
    }
}

/// Minimum cells per scan raster: fine enough to rank feature sizes.
pub const FLOOD_CELLS: u64 = 1 << 18;
/// Scan escape cap: a count at or above it is inside the set.
pub const SCAN_MAX_ITER: u32 = 255;
/// Counts below this are blocked: the whole fractal's outermost shell.
pub const MIN_ESCAPE: u32 = 1;
/// Seeds per side of an area's first seed lattice.
pub const SEED_GRID: u32 = 128;
/// A feature spans counts `b ..= b + SHELLS`.
pub const SHELLS: u32 = 5;
/// Fiord depth margin, in cells: a fiord reaches further than this inside its
/// hull.
pub const REACH: u32 = 1;
/// Each ladder step keeps this share of every edge's distance to the leaf.
pub const LADDER_RATIO: Ratio = Ratio::new(7, 10);
/// The ladder stops once the leaf rectangle exceeds this share of the area.
pub const LEAF_SHARE: Ratio = Ratio::new(1, 20);
/// Features kept per ladder area, largest first.
pub const KEEP: u32 = 200;
/// Minimum cells across a candidate's bounding box (its shorter side).
pub const MIN_CELLS: u32 = 12;
/// Cost of zooming against panning in the zoom distance.
pub const RHO: Ratio = Ratio::new(1, 1);
/// A hop of distance `h` costs `h^BADNESS_EXPONENT + BADNESS_CHARGE`.
pub const BADNESS_EXPONENT: u32 = 2;
/// Cost per hop.
pub const BADNESS_CHARGE: Ratio = Ratio::new(1, 1);
/// Fractional bits of distances and costs.
pub const METRIC_FRAC_BITS: u32 = 32;

/// Largest `metric_frac_bits` the metric's 128-bit arithmetic holds: a base-2
/// logarithm of a 256-bit value is below 2⁸ (`LOG2_INT_BITS`), doubled (one
/// bit), times ln 2 in Q64, must stay below 2¹²⁸.
pub const METRIC_FRAC_BITS_MAX: u32 = 128 - 64 - 1 - LOG2_INT_BITS;
/// Integer bits of the base-2 logarithm of a 256-bit value.
pub const LOG2_INT_BITS: u32 = 8;

/// Every number the landmark path depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LandmarkParams {
    pub flood_cells: u64,
    pub scan_max_iter: u32,
    pub min_escape: u32,
    pub seed_grid: u32,
    pub shells: u32,
    pub reach: u32,
    pub ladder_ratio: Ratio,
    pub leaf_share: Ratio,
    pub keep: u32,
    pub min_cells: u32,
    pub rho: Ratio,
    pub badness_exponent: u32,
    pub badness_charge: Ratio,
    pub metric_frac_bits: u32,
    /// Flood cap of the leaf's canonical island.
    pub canonical_max_flood_points: u64,
}

impl Default for LandmarkParams {
    fn default() -> Self {
        LandmarkParams {
            flood_cells: FLOOD_CELLS,
            scan_max_iter: SCAN_MAX_ITER,
            min_escape: MIN_ESCAPE,
            seed_grid: SEED_GRID,
            shells: SHELLS,
            reach: REACH,
            ladder_ratio: LADDER_RATIO,
            leaf_share: LEAF_SHARE,
            keep: KEEP,
            min_cells: MIN_CELLS,
            rho: RHO,
            badness_exponent: BADNESS_EXPONENT,
            badness_charge: BADNESS_CHARGE,
            metric_frac_bits: METRIC_FRAC_BITS,
            canonical_max_flood_points: CANONICAL_MAX_FLOOD_POINTS,
        }
    }
}

impl LandmarkParams {
    /// Whether the path can be computed with these values: rasters and seed
    /// lattices are non-empty, the ladder shrinks, every hop costs more than
    /// zero, and the metric's arithmetic fits.
    pub fn is_valid(&self) -> bool {
        self.flood_cells > 0
            && self.scan_max_iter > self.min_escape
            && self.seed_grid > 0
            && self.ladder_ratio.den > 0
            && self.ladder_ratio.num < self.ladder_ratio.den
            && self.leaf_share.den > 0
            && self.leaf_share.num > 0
            && self.rho.num > 0
            && self.rho.den > 0
            && self.badness_exponent > 0
            && self.badness_charge.den > 0
            && self.metric_frac_bits <= METRIC_FRAC_BITS_MAX
            // The charge is at least one unit, so every cost is positive.
            && ((self.badness_charge.num as u128) << self.metric_frac_bits)
                >= self.badness_charge.den as u128
            && self.canonical_max_flood_points > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        assert!(LandmarkParams::default().is_valid());
    }

    #[test]
    fn invalid_values_are_refused() {
        let ok = LandmarkParams::default();
        let cases = [
            LandmarkParams { seed_grid: 0, ..ok },
            LandmarkParams {
                ladder_ratio: Ratio::new(1, 1),
                ..ok
            },
            LandmarkParams {
                badness_charge: Ratio::new(0, 1),
                ..ok
            },
            LandmarkParams {
                rho: Ratio::new(1, 0),
                ..ok
            },
            LandmarkParams {
                metric_frac_bits: METRIC_FRAC_BITS_MAX + 1,
                ..ok
            },
        ];
        for p in cases {
            assert!(!p.is_valid(), "{p:?}");
        }
    }
}
