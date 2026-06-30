/// Viewport leaf-area enumeration.
///
/// Given a view (an origin/step pixel grid over fractal space), determine which
/// distinct *canonical leaf areas* are present, so the renderer can draw the
/// canonical island of each.  The algorithm samples the view on an N×N grid and
/// uses a single, uniform exclusion list to skip whole regions once they are
/// classified:
///
/// 1. For each sampled pixel (every `scan_step` pixels, row-major):
///    1. If the pixel falls inside any already-excluded area, skip it.
///    2. Otherwise decode the point (`bisect::decode_locate`).
///    3. If it is in a contracted-away (dead) region, add that whole sliver to
///       the exclusion list and move on.
///    4. If it is in a leaf area not seen before, register it.
///    5. If registering would exceed `max_leaves`, abort with `TooMany`
///       ("too many leaf areas — increase zoom").
///    6. Either way, add the leaf rectangle to the exclusion list so the rest
///       of that leaf is skipped, then move on.
/// 2. Return the list of registered leaf areas.
///
/// Both dead slivers and processed leaf rectangles share the same exclusion
/// list, so every later sample landing in either is skipped cheaply.

use crate::bisect::{decode_locate, Located};
use crate::discovery::DiscoveryParams;
use crate::fixed::{Fixed, Rect};
use std::collections::HashSet;

/// A distinct leaf area found within the view.
#[derive(Clone, Debug)]
pub struct LeafArea {
    /// The leaf rectangle (the final contracted rect of the bisection).
    pub rect: Rect,
    /// The bisection path — the canonical identity of this leaf area.
    pub path: String,
}

/// Result of enumerating the leaf areas in a view.
#[derive(Debug)]
pub enum LeafEnumOutcome {
    /// The distinct leaf areas present in the view (at most `max_leaves`).
    Leaves(Vec<LeafArea>),
    /// More than `max_leaves` distinct leaf areas are present; the caller
    /// should ask the user to zoom in.  Carries the cap that was exceeded.
    TooMany { max: usize },
    /// The decode budget (`max_decodes`) was hit before the scan finished.
    /// This is the zoom-out guard: when little is excluded, the scan would
    /// otherwise decode a huge grid at hundreds of ms per point.  The view is
    /// too dense/dead to enumerate here — the caller should ask the user to
    /// zoom in.  Carries the number of decodes performed.
    BudgetExhausted { decodes: usize },
}

/// Enumerate the distinct canonical leaf areas visible in a view.
///
/// The view is described the same way as the raster renderer
/// (`ffi::bs_render_viewport`): pixel `(col, row)` maps to fractal coordinate
/// `(origin_re + col*step, origin_im + row*step)`.  Sampling steps by
/// `scan_step` pixels on both axes (clamped to ≥ 1).  The decode parameters
/// (`initial_area`, `params`, `rng_seed`, `num_bits`, `o`, `p`, `q`,
/// `path_prefix`) are exactly those used elsewhere for decoding.
///
/// `max_decodes` bounds the total number of (non-excluded) decodes — the
/// zoom-out guard.  A decode is ~hundreds of ms, so at zoom-out, where almost
/// nothing is excluded, an unbounded scan would decode the whole grid and hang.
/// On hitting the budget the scan aborts with [`LeafEnumOutcome::BudgetExhausted`].
/// `0` disables the budget (unbounded).
#[allow(clippy::too_many_arguments)]
pub fn enumerate_leaf_areas(
    origin_re: f64,
    origin_im: f64,
    step: f64,
    width_px: u32,
    height_px: u32,
    scan_step: u32,
    max_leaves: usize,
    max_decodes: usize,
    initial_area: Rect,
    params: &DiscoveryParams,
    rng_seed: u64,
    num_bits: usize,
    o: u64,
    p: u64,
    q: u64,
    path_prefix: &str,
) -> LeafEnumOutcome {
    let scan = scan_step.max(1) as usize;

    // Uniform exclusion list: processed leaf rects AND dead slivers.
    let mut exclusions: Vec<Rect> = Vec::new();
    let mut leaves: Vec<LeafArea> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    // Count of (non-excluded) decodes performed — bounded by `max_decodes`.
    let mut decodes: usize = 0;

    for row in (0..height_px).step_by(scan) {
        let im = Fixed::from_f64(origin_im + row as f64 * step);
        for col in (0..width_px).step_by(scan) {
            let re = Fixed::from_f64(origin_re + col as f64 * step);

            // 1.i — already covered by a leaf or a dead region.
            if exclusions.iter().any(|r| r.contains(re, im)) {
                continue;
            }

            // Zoom-out guard: abort once the decode budget is hit, rather than
            // decode the whole (barely-excluded) grid at hundreds of ms each.
            if max_decodes != 0 && decodes >= max_decodes {
                return LeafEnumOutcome::BudgetExhausted { decodes };
            }
            decodes += 1;

            // 1.ii — decode (the bisection algorithm) and classify.
            match decode_locate(
                re, im, num_bits, initial_area, params, rng_seed, o, p, q, path_prefix,
            ) {
                // 1.iii — contracted-away region: exclude the whole sliver.
                Located::Dead { rect } => {
                    if let Some(r) = rect {
                        exclusions.push(r);
                    }
                    // If the sliver could not be determined we simply advance to
                    // the next sample (it gets re-decoded, but that is rare).
                }
                // 1.iv / 1.vi — a leaf area.
                Located::Leaf { rect, path } => {
                    if !seen.contains(&path) {
                        // 1.v — soft cap on the number of distinct leaf areas.
                        if leaves.len() >= max_leaves {
                            return LeafEnumOutcome::TooMany { max: max_leaves };
                        }
                        seen.insert(path.clone());
                        leaves.push(LeafArea { rect, path });
                    }
                    // 1.vi — exclude the leaf so the rest of it is skipped.
                    exclusions.push(rect);
                }
            }
        }
    }

    LeafEnumOutcome::Leaves(leaves)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol;

    // A shallow bisection depth keeps leaf areas large enough to be hit by a
    // coarse sample grid.  At the full 32-bit encode depth, contraction carves
    // the live (leaf) region far below sample size, so a coarse scan over a
    // shallow view sees almost only contracted-away regions — leaf areas only
    // become sample-visible near their (very deep) canonical zoom.
    const SHALLOW_BITS: usize = 4;

    // Scan the full encode area on a 64×64 grid, sampling every 4 px.
    // No decode budget (0) — these shallow-depth scans finish quickly.
    fn enumerate_full_encode(max_leaves: usize, num_bits: usize) -> LeafEnumOutcome {
        enumerate_leaf_areas(
            -2.5, -2.0, 4.0 / 64.0, 64, 64, 4, max_leaves, 0,
            protocol::encode_area(), &protocol::encode_params(),
            protocol::ENCODE_RNG_SEED, num_bits, 0, 0, 0, "O",
        )
    }

    fn leaves_of(outcome: LeafEnumOutcome) -> Vec<LeafArea> {
        match outcome {
            LeafEnumOutcome::Leaves(l) => l,
            other => panic!("unexpected non-Leaves outcome: {other:?}"),
        }
    }

    #[test]
    fn decode_budget_aborts_at_full_depth() {
        // Full encode area at the real 32-bit depth: almost every sample is a
        // fresh dead/leaf region (little gets excluded), so a small decode
        // budget must abort instead of scanning the whole grid.
        let out = enumerate_leaf_areas(
            -2.5, -2.0, 4.0 / 64.0, 64, 64, 4, 1000, /*max_decodes*/ 5,
            protocol::encode_area(), &protocol::encode_params(),
            protocol::ENCODE_RNG_SEED, protocol::BITS_PER_POINT as usize,
            0, 0, 0, "O",
        );
        assert!(
            matches!(out, LeafEnumOutcome::BudgetExhausted { decodes } if decodes == 5),
            "expected BudgetExhausted(5), got {out:?}",
        );
    }

    #[test]
    fn enumerate_is_deterministic() {
        let a = leaves_of(enumerate_full_encode(10_000, SHALLOW_BITS));
        let b = leaves_of(enumerate_full_encode(10_000, SHALLOW_BITS));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.path, y.path, "leaf order/identity must be deterministic");
            assert_eq!(x.rect.re_min.0, y.rect.re_min.0);
            assert_eq!(x.rect.re_max.0, y.rect.re_max.0);
            assert_eq!(x.rect.im_min.0, y.rect.im_min.0);
            assert_eq!(x.rect.im_max.0, y.rect.im_max.0);
        }
    }

    #[test]
    fn registered_leaf_paths_are_distinct() {
        let leaves = leaves_of(enumerate_full_encode(10_000, SHALLOW_BITS));
        assert!(leaves.len() >= 2, "view should contain multiple leaf areas");
        let unique: HashSet<&String> = leaves.iter().map(|l| &l.path).collect();
        assert_eq!(unique.len(), leaves.len(), "leaf paths must be unique");
    }

    #[test]
    fn cap_triggers_too_many_relative_to_actual_count() {
        // Discover the actual number of leaf areas with a generous cap...
        let n = leaves_of(enumerate_full_encode(10_000, SHALLOW_BITS)).len();
        assert!(n >= 2, "need at least two leaves to exercise the cap");

        // ...a cap below it must report TooMany...
        let under = enumerate_full_encode(n - 1, SHALLOW_BITS);
        assert!(
            matches!(under, LeafEnumOutcome::TooMany { max } if max == n - 1),
            "cap {} (< {n}) should report TooMany, got {under:?}", n - 1,
        );

        // ...and a cap equal to it must return exactly that many leaves.
        let exact = leaves_of(enumerate_full_encode(n, SHALLOW_BITS));
        assert_eq!(exact.len(), n, "cap == count should return all leaves");
    }

    #[test]
    fn scan_step_is_clamped_to_at_least_one() {
        // scan_step = 0 must not hang/panic; it is clamped to 1.
        let out = enumerate_leaf_areas(
            -2.5, -2.0, 4.0 / 16.0, 16, 16, 0, 10_000, 0,
            protocol::encode_area(), &protocol::encode_params(),
            protocol::ENCODE_RNG_SEED, SHALLOW_BITS, 0, 0, 0, "O",
        );
        assert!(matches!(out, LeafEnumOutcome::Leaves(_)));
    }
}
