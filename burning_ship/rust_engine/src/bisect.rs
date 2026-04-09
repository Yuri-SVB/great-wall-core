/// Bisection tree: maps a bit sequence to a fractal location.
///
/// At each level:
/// 1. Discover islands in the current rectangle
/// 2. Compute the weighted median of barycenters (weighted by log2(total_area/flood_area))
/// 3. Split along the longer side at the weighted median
/// 4. Consume a seed bit to choose a child
/// 5. Contract the larger child if selected

use crate::fixed::{Fixed, Rect};
use crate::discovery::{discover_islands, DiscoveryParams, InheritedSeed, Island};

/// Minimum integer weight for an island in weighted median (avoids zero-weight entries).
const MIN_ISLAND_WEIGHT: u128 = 1;

/// Contraction formula coefficients: f(r) = (larger + CONT_MUL * smaller) / (CONT_DIV * larger).
const CONTRACTION_MULTIPLIER: u64 = 3;
const CONTRACTION_DIVISOR: u64 = 4;

/// Compute the weighted median of island barycenters along the split axis.
///
/// Weight for island j: score = log2(good_total_area / flood_area_j) as
/// fixed-point u128 with LOG2_FRAC_BITS fractional bits.
/// Rarer (smaller) islands have higher scores and more influence.
///
/// The weighted median is the coordinate where the cumulative weight
/// from below reaches half the total weight.  All arithmetic is integer.
fn weighted_median_fixed(islands: &[Island], vertical: bool) -> Fixed {
    if islands.is_empty() {
        return Fixed::ZERO;
    }
    if islands.len() == 1 {
        return islands[0].center_coord(vertical);
    }

    // Build (coord, weight) pairs, sorted by coord.
    let mut entries: Vec<(i64, u128)> = islands.iter()
        .map(|i| (i.center_coord(vertical).0, i.score.max(MIN_ISLAND_WEIGHT)))
        .collect();
    entries.sort_by_key(|&(coord, _)| coord);

    let total_weight: u128 = entries.iter().map(|&(_, w)| w).sum();
    let half = total_weight / 2;

    // Walk from the low end accumulating weight until we cross the midpoint.
    let mut cumulative: u128 = 0;
    for &(coord, weight) in &entries {
        cumulative += weight;
        if cumulative >= half {
            return Fixed(coord);
        }
    }

    // Fallback: last entry (shouldn't reach here).
    Fixed(entries.last().unwrap().0)
}

/// Apply contraction: new_dist = old_dist * num / den (in i128).
fn apply_contraction(dist: i64, num: u64, den: u64) -> i64 {
    if den == 0 {
        return dist;
    }
    let wide = dist as i128 * num as i128;
    (wide / den as i128) as i64
}

/// Result of encoding: the final rectangle and the path taken.
#[derive(Clone, Debug)]
pub struct EncodeResult {
    /// Final rectangle after all bisections.
    pub final_rect: Rect,
    /// Center of the final rectangle (the encoded point).
    pub point_re: Fixed,
    pub point_im: Fixed,
    /// Number of bits actually consumed.
    pub bits_consumed: usize,
    /// Log of each bisection step for visualization/debugging.
    pub steps: Vec<BisectStep>,
    /// Bisection path string: "O" followed by one direction letter per level.
    /// Letters: l/L (left), r/R (right), u/U (up), d/D (down).
    /// Lowercase = smaller half, uppercase = larger half.
    pub path: String,
    /// Inherited seeds for the next bisection level (carried forward for
    /// incremental encoding).
    pub inherited_seeds: Vec<InheritedSeed>,
}

/// Record of a single bisection step (for debugging/visualization).
#[derive(Clone, Debug)]
pub struct BisectStep {
    pub area: Rect,
    pub split_coord: Fixed,
    pub split_vertical: bool,
    pub bit: u8,
    pub chose_larger: bool,
    pub contraction_num: u64,
    pub contraction_den: u64,
    pub islands_found: usize,
    /// Barycenters and pixel counts of discovered good islands (for visualization).
    pub island_centers: Vec<(Fixed, Fixed, u64)>,
}

/// Core bisection logic shared by encode and decode.
///
/// For encode: bits are consumed from the input.
/// For decode: bits are determined by point membership.
///
/// Returns (final_rect, steps, bits, valid, path, inherited_seeds).
/// `valid` is true for encode (always valid) or when the decode point stays
/// inside every contracted child.  It is false when the point falls in a
/// region that was excluded by contraction of the larger half.
#[allow(unused_variables)]
fn bisect_core(
    initial_area: Rect,
    params: &DiscoveryParams,
    rng_seed: u64,
    num_bits: usize,
    // If Some(bits), we are encoding and consume from this.
    // If None, we are decoding and determine bits from the point.
    input_bits: Option<&[u8]>,
    // For decode: the point to locate.
    decode_point: Option<(Fixed, Fixed)>,
    // Perturbation parameters (orbit seed o, additive shift p, linear term q).
    o: u64,
    p: u64,
    q: u64,
    // Path prefix: accumulated path from prior points / Argon2 marker.
    // First point of a pipeline should pass "O".
    path_prefix: &str,
    initial_inherited_seeds: Vec<InheritedSeed>,
) -> (Rect, Vec<BisectStep>, Vec<u8>, bool, String, Vec<InheritedSeed>) {
    let mut current = initial_area;
    let mut steps = Vec::with_capacity(num_bits);
    let mut output_bits = Vec::with_capacity(num_bits);
    let mut valid = true;
    // Path string: start from prefix, append one direction letter per level.
    let mut path = String::with_capacity(path_prefix.len() + num_bits);
    path.push_str(path_prefix);
    // Inherited seeds for the next level.
    let mut inherited_seeds: Vec<InheritedSeed> = initial_inherited_seeds;

    log_verbose!("[bisect] start: {} bits, area re=[{:.6},{:.6}] im=[{:.6},{:.6}]",
              num_bits,
              initial_area.re_min.to_f64(), initial_area.re_max.to_f64(),
              initial_area.im_min.to_f64(), initial_area.im_max.to_f64());

    for bit_index in 0..num_bits {
        // Discover islands — PRNG seeded from path string, not rect coords.
        // Inherited seeds from the parent level are re-flood-filled first.
        let (islands, good_points) = discover_islands(&current, params, rng_seed, o, p, q, &path, &inherited_seeds);

        let (split_coord, split_vertical) = if islands.is_empty() {
            // Fallback: geometric bisection at center
            let (cre, cim) = current.center();
            let sv = current.is_wider();
            log_verbose!("[bisect] bit {}: FALLBACK (0 islands) → geometric center \
                      split_{}={:.6}  area={:.6}x{:.6}",
                      bit_index,
                      if sv { "re" } else { "im" },
                      if sv { cre.to_f64() } else { cim.to_f64() },
                      current.width().to_f64(), current.height().to_f64());
            if sv { (cre, true) } else { (cim, false) }
        } else {
            let sv = current.is_wider();
            let bary = weighted_median_fixed(&islands, sv);

            // Per-island detail: position along split axis, pixel count, score
            // for (idx, isl) in islands.iter().enumerate() {
            //     let coord = if sv { isl.center_re.to_f64() } else { isl.center_im.to_f64() };
            //     log_verbose!("[bisect]   island {}: {}={:.6} esc={} px={} score={} bbox={:.6}x{:.6}",
            //               idx,
            //               if sv { "re" } else { "im" },
            //               coord,
            //               isl.escape_count,
            //               isl.pixel_count,
            //               isl.score,
            //               (isl.bbox.re_max.0 - isl.bbox.re_min.0) as f64 / (1u64 << 60) as f64,
            //               (isl.bbox.im_max.0 - isl.bbox.im_min.0) as f64 / (1u64 << 60) as f64);
            // }

            (bary, sv)
        };

        // Create children
        let (child_lo, child_hi) = if split_vertical {
            (
                Rect {
                    re_min: current.re_min,
                    re_max: split_coord,
                    im_min: current.im_min,
                    im_max: current.im_max,
                },
                Rect {
                    re_min: split_coord,
                    re_max: current.re_max,
                    im_min: current.im_min,
                    im_max: current.im_max,
                },
            )
        } else {
            (
                Rect {
                    re_min: current.re_min,
                    re_max: current.re_max,
                    im_min: current.im_min,
                    im_max: split_coord,
                },
                Rect {
                    re_min: current.re_min,
                    re_max: current.re_max,
                    im_min: split_coord,
                    im_max: current.im_max,
                },
            )
        };

        // Identify larger/smaller by span along the split axis (Fixed, no f64).
        let lo_span = if split_vertical {
            split_coord.0 - current.re_min.0
        } else {
            split_coord.0 - current.im_min.0
        };
        let hi_span = if split_vertical {
            current.re_max.0 - split_coord.0
        } else {
            current.im_max.0 - split_coord.0
        };
        let larger_is_hi = hi_span >= lo_span;

        // Determine the bit
        //
        // Bit semantics (directional, not size-based):
        //   0 = left  (lo re) for vertical splits, up   (hi im) for horizontal splits
        //   1 = right (hi re) for vertical splits, down (lo im) for horizontal splits
        let bit: u8 = if let Some(bits) = input_bits {
            // Encoding: consume from input
            bits[bit_index]
        } else {
            // Decoding: determine by point membership
            let (pre, pim) = decode_point.unwrap();
            let point_in_hi = if split_vertical {
                pre.0 >= split_coord.0
            } else {
                pim.0 >= split_coord.0
            };
            // vertical: hi=right=1, lo=left=0
            // horizontal: hi=up=0, lo=down=1
            if split_vertical {
                if point_in_hi { 1 } else { 0 }
            } else {
                if point_in_hi { 0 } else { 1 }
            }
        };

        output_bits.push(bit);

        // Map bit to child: 0→left/up, 1→right/down
        let choose_hi = if split_vertical { bit == 1 } else { bit == 0 };
        let mut chosen = if choose_hi { child_hi } else { child_lo };
        let chose_larger = if larger_is_hi { choose_hi } else { !choose_hi };

        // Append direction to path string.
        // Letter: l/r for vertical, u/d for horizontal.
        // Case: lowercase = smaller half, UPPERCASE = larger half.
        let dir_char = match (split_vertical, choose_hi, chose_larger) {
            (true,  false, false) => 'l', // left, smaller
            (true,  false, true)  => 'L', // left, larger
            (true,  true,  false) => 'r', // right, smaller
            (true,  true,  true)  => 'R', // right, larger
            (false, true,  false) => 'u', // up (hi im), smaller
            (false, true,  true)  => 'U', // up (hi im), larger
            (false, false, false) => 'd', // down (lo im), smaller
            (false, false, true)  => 'D', // down (lo im), larger
        };
        path.push(dir_char);

        // Compute contraction using island pixel counts
        // For the contraction we need area proxies from the islands in each child.
        // Since all islands share the same discovery, we use the total pixel counts
        // as the "area" measure. But actually, the contraction formula is about
        // the child rectangles' areas, not island areas. The children's spans are
        // lo_span and hi_span (in Fixed, along split axis). The perpendicular
        // dimension is the same for both. So the area ratio = span ratio.
        let (cont_num, cont_den) = if chose_larger {
            // r = smaller_span / larger_span
            // f(r) = (1 + 3r) / 4 = (larger + 3*smaller) / (4 * larger)
            let s = lo_span.min(hi_span) as u64;
            let l = lo_span.max(hi_span) as u64;
            if l == 0 {
                (1u64, 1u64)
            } else {
                let num = l + CONTRACTION_MULTIPLIER * s;
                let den = CONTRACTION_DIVISOR * l;
                (num, den)
            }
        } else {
            (1u64, 1u64)
        };

        // Apply contraction if larger is chosen.
        if chose_larger {
            if split_vertical {
                if larger_is_hi {
                    let dist = chosen.re_max.0 - split_coord.0;
                    let new_dist = apply_contraction(dist, cont_num, cont_den);
                    chosen.re_max = Fixed(split_coord.0 + new_dist);
                } else {
                    let dist = split_coord.0 - chosen.re_min.0;
                    let new_dist = apply_contraction(dist, cont_num, cont_den);
                    chosen.re_min = Fixed(split_coord.0 - new_dist);
                }
            } else {
                if larger_is_hi {
                    let dist = chosen.im_max.0 - split_coord.0;
                    let new_dist = apply_contraction(dist, cont_num, cont_den);
                    chosen.im_max = Fixed(split_coord.0 + new_dist);
                } else {
                    let dist = split_coord.0 - chosen.im_min.0;
                    let new_dist = apply_contraction(dist, cont_num, cont_den);
                    chosen.im_min = Fixed(split_coord.0 - new_dist);
                }
            }
        }

        // After contraction, check if the decode point is still inside.
        if let Some((pre, pim)) = decode_point {
            if valid && !chosen.contains(pre, pim) {
                log_verbose!("[bisect] bit {}: decode point ({:.6},{:.6}) OUTSIDE contracted child → invalid",
                          bit_index, pre.to_f64(), pim.to_f64());
                valid = false;
            }
        }

        let dir_label = if split_vertical {
            if choose_hi { "right" } else { "left" }
        } else {
            if choose_hi { "up" } else { "down" }
        };
        log_verbose!("[bisect] bit {}: chose={}{} contraction={}/{} → child re=[{:.6},{:.6}] im=[{:.6},{:.6}]",
                  bit_index, dir_label,
                  if chose_larger { "(larger)" } else { "(smaller)" },
                  cont_num, cont_den,
                  chosen.re_min.to_f64(), chosen.re_max.to_f64(),
                  chosen.im_min.to_f64(), chosen.im_max.to_f64());

        // Build inherited seeds for the next level: all flood-filled points
        // from good islands that fall inside the chosen (possibly contracted)
        // child.  Using all points (not just barycenters) handles non-convex
        // islands whose barycenter may lie outside the island itself.
        let prev_count = good_points.len();
        inherited_seeds = good_points.into_iter()
            .filter(|seed| chosen.contains(seed.re, seed.im))
            .collect();

        log_verbose!("[bisect] bit {}: {} of {} good points inherited by child",
                  bit_index, inherited_seeds.len(), prev_count);

        let island_centers: Vec<(Fixed, Fixed, u64)> = islands.iter()
            .map(|isl| (isl.center_re, isl.center_im, isl.pixel_count))
            .collect();

        steps.push(BisectStep {
            area: current,
            split_coord,
            split_vertical,
            bit,
            chose_larger,
            contraction_num: cont_num,
            contraction_den: cont_den,
            islands_found: islands.len(),
            island_centers,
        });

        current = chosen;
    }

    log_verbose!("[bisect] final rect re=[{:.6},{:.6}] im=[{:.6},{:.6}] ({:.6}x{:.6}) path={}",
              current.re_min.to_f64(), current.re_max.to_f64(),
              current.im_min.to_f64(), current.im_max.to_f64(),
              current.width().to_f64(), current.height().to_f64(),
              path);

    (current, steps, output_bits, valid, path, inherited_seeds)
}

/// Encode a bit sequence into a fractal location.
///
/// `path_prefix` is the accumulated path from prior points in the pipeline.
/// First point should pass "O". Subsequent points pass the previous point's
/// full path (with any Argon2 markers appended between stages).
pub fn encode(
    seed_bits: &[u8],
    initial_area: Rect,
    params: &DiscoveryParams,
    rng_seed: u64,
    o: u64,
    p: u64,
    q: u64,
    path_prefix: &str,
) -> EncodeResult {
    encode_with_seeds(seed_bits, initial_area, params, rng_seed, o, p, q, path_prefix, Vec::new())
}

/// Encode with explicit initial inherited_seeds (for incremental encoding).
pub fn encode_with_seeds(
    seed_bits: &[u8],
    initial_area: Rect,
    params: &DiscoveryParams,
    rng_seed: u64,
    o: u64,
    p: u64,
    q: u64,
    path_prefix: &str,
    initial_seeds: Vec<InheritedSeed>,
) -> EncodeResult {
    let (final_rect, steps, _, _valid, path, inherited_seeds) = bisect_core(
        initial_area, params, rng_seed, seed_bits.len(),
        Some(seed_bits), None, o, p, q, path_prefix, initial_seeds,
    );

    let (cre, cim) = final_rect.center();
    EncodeResult {
        final_rect,
        point_re: cre,
        point_im: cim,
        bits_consumed: seed_bits.len(),
        steps,
        path,
        inherited_seeds,
    }
}

/// Decode a point back to bits.
///
/// `path_prefix` must match the prefix used during encoding for this point.
pub fn decode(
    point_re: Fixed,
    point_im: Fixed,
    num_bits: usize,
    initial_area: Rect,
    params: &DiscoveryParams,
    rng_seed: u64,
    o: u64,
    p: u64,
    q: u64,
    path_prefix: &str,
) -> Vec<u8> {
    let (_, _, bits, _valid, _path, _seeds) = bisect_core(
        initial_area, params, rng_seed, num_bits,
        None, Some((point_re, point_im)), o, p, q, path_prefix, Vec::new(),
    );
    bits
}

/// Decode a point back to bits, also returning the final leaf rectangle and
/// whether the point is valid (i.e. stayed inside every contracted child).
pub fn decode_full(
    point_re: Fixed,
    point_im: Fixed,
    num_bits: usize,
    initial_area: Rect,
    params: &DiscoveryParams,
    rng_seed: u64,
    o: u64,
    p: u64,
    q: u64,
    path_prefix: &str,
) -> (Vec<u8>, Rect, bool, String, Vec<BisectStep>) {
    let (final_rect, steps, bits, valid, path, _seeds) = bisect_core(
        initial_area, params, rng_seed, num_bits,
        None, Some((point_re, point_im)), o, p, q, path_prefix, Vec::new(),
    );
    (bits, final_rect, valid, path, steps)
}
