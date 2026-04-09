/// Island discovery via random sampling + flood fill.
///
/// Finds connected regions of equal escape count within a rectangular area.

use crate::fixed::{Fixed, Rect};
use crate::fractal::escape_count_generic;
use std::collections::{BTreeSet, HashSet, VecDeque};

/// Minimum pixel count for a flood fill to qualify as a good island.
/// Islands below this threshold are noise — their points are registered
/// in the collision store (excluded area) but not added to the island list.
const MIN_ISLAND_PIXEL_COUNT: u64 = 1;

/// Minimum escape count to consider an island structurally interesting.
/// Empirically, the first meaningful islands appear around escape count ~17.
const MIN_ESCAPE_COUNT_THRESHOLD: u32 = 10;

/// Number of fractional bits in the fixed-point log2 score.
const LOG2_FRAC_BITS: u32 = 16;

/// Maximum score assigned to degenerate islands (zero flood area).
/// Equivalent to log2 score of 64 in the fixed-point representation.
const MAX_DEGENERATE_SCORE: u128 = 64 << LOG2_FRAC_BITS;

/// Exclusion threshold denominator (fraction = num / EXCLUSION_DENOM).
const EXCLUSION_DENOM: u128 = 256;

/// Direction bits for flood-fill collision testing.
const DIR_LEFT: u8 = 1;
const DIR_RIGHT: u8 = 2;
const DIR_DOWN: u8 = 4;
const DIR_UP: u8 = 8;
const DIR_ALL: u8 = DIR_LEFT | DIR_RIGHT | DIR_DOWN | DIR_UP;

/// A discovered island: a connected region with uniform escape count.
#[derive(Clone, Debug)]
pub struct Island {
    /// Barycenter of the flood-filled region (in Fixed).
    pub center_re: Fixed,
    pub center_im: Fixed,
    /// Measured area as pixel count (integer).
    pub pixel_count: u64,
    /// Pixel step size used during flood fill (in Fixed).
    pub pixel_delta: Fixed,
    /// Escape count for all points in this island.
    pub escape_count: u32,
    /// Bounding box of the island.
    pub bbox: Rect,
    /// Score = log2(good_total_area / flood_area) as fixed-point u128 with
    /// LOG2_FRAC_BITS fractional bits.  Higher means rarer (more informative).
    /// Computed after discovery; 0 until set.
    pub score: u128,
}

impl Island {
    /// Barycenter coordinate along the split axis (re if vertical, im if horizontal).
    pub fn center_coord(&self, vertical: bool) -> Fixed {
        if vertical { self.center_re } else { self.center_im }
    }
}

/// Parameters for island discovery.
#[derive(Clone, Debug)]
pub struct DiscoveryParams {
    pub max_iter: u32,
    /// Minimum number of grid cells covering the generation area.
    /// p0 is the largest power-of-two pixel size such that
    /// (width * height) / p0² >= min_grid_cells.
    pub min_grid_cells: u64,
    /// Maximum flood-fill step as a power-of-two divisor of area width.
    /// pixel_size = area_width >> p_max_shift.
    pub p_max_shift: u32,
    /// Maximum number of points in a single flood fill.
    pub max_flood_points: u64,
    /// Target number of good islands to find.
    pub target_good: u32,
    /// Stop when this fraction of area is excluded (as numerator over 256).
    pub exclusion_threshold_num: u32,
    /// Maximum random sample attempts.
    pub max_attempts: u64,
}

impl Default for DiscoveryParams {
    fn default() -> Self {
        DiscoveryParams {
            max_iter: 128,
            min_grid_cells: 4096, // p0 ≈ sqrt(area / 4096)
            p_max_shift: 3,       // ~1/8 of area width ≈ 0.125
            max_flood_points: 50_000,
            target_good: 100,
            exclusion_threshold_num: 230, // 230/256 ≈ 0.898
            max_attempts: 50_000,
        }
    }
}

/// Deterministic point generation for island discovery.
///
/// Architecture:
///   area_seed  = hash(o, p, q, path)   — one hash per bisection level
///   (re, im)   = hash(area_seed, i)    — one hash per candidate point
///
/// All points within a bisection level are derivable independently from
/// (area_seed, index), enabling future parallel batch generation.
pub struct PointGen {
    /// 128-bit area seed: encodes the full fractal context for this level.
    seed_lo: u64,
    seed_hi: u64,
}

/// Wyhash domain-separation salts.  These are the first eight 64-bit chunks
/// of the fractional expansion of e (Euler's number), ensuring no
/// structural relationship between the XOR masks.
const WYMIX_SALT_0: u64 = 0xb7e151628aed2a6a;  // frac(e) bits 0..63
const WYMIX_SALT_1: u64 = 0xbf7158809cf4f3c7;  // frac(e) bits 64..127
const WYMIX_SALT_2: u64 = 0x62e7160f38b4da56;  // frac(e) bits 128..191
const WYMIX_SALT_3: u64 = 0xa784d9045190cfef;  // frac(e) bits 192..255
const WYMIX_SALT_4: u64 = 0x324e7738926cfbe5;  // frac(e) bits 256..319
const WYMIX_SALT_5: u64 = 0xf4bf8d8d8c31d763;  // frac(e) bits 320..383
const WYMIX_SALT_6: u64 = 0xda06c80abb1185eb;  // frac(e) bits 384..447
const WYMIX_SALT_7: u64 = 0x4f7c7b5757f59584;  // frac(e) bits 448..511

/// Point-derivation salts.  These are the first two 64-bit chunks of the
/// fractional expansion of sqrt(2).
const DERIVE_SALT_RE: u64 = 0x6a09e667f3bcc908;  // frac(sqrt(2)) bits 0..63
const DERIVE_SALT_IM: u64 = 0xbb67ae8584caa73b;  // frac(sqrt(2)) bits 64..127

impl PointGen {
    /// Wyhash-style 64-bit mixing: multiply two 64-bit values, XOR the
    /// high and low halves of the 128-bit product.
    #[inline]
    fn wymix(a: u64, b: u64) -> u64 {
        let r = a as u128 * b as u128;
        (r as u64) ^ ((r >> 64) as u64)
    }

    /// Hash the path string into a 64-bit digest using splitmix64.
    fn hash_path(path: &str, rng_seed: u64) -> u64 {
        let mut h = rng_seed;
        for &b in path.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_add(0x9E3779B97F4A7C15);
            h ^= h >> 30;
            h = h.wrapping_mul(0xBF58476D1CE4E5B9);
            h ^= h >> 27;
            h = h.wrapping_mul(0x94D049BB133111EB);
            h ^= h >> 31;
        }
        h
    }

    /// Build a PointGen from the fractal parameters and bisection path.
    ///
    /// Both seed halves depend on all four inputs (o, p, q, path_hash)
    /// via independent mixing orders, ensuring every input bit
    /// pseudorandomly affects both Re and Im coordinates.
    ///
    /// seed_lo = wymix(wymix(o ^ s0, p ^ s1), wymix(q ^ s2, path_hash ^ s3))
    /// seed_hi = wymix(wymix(o ^ s4, q ^ s5), wymix(p ^ s6, path_hash ^ s7))
    pub fn new(o: u64, p: u64, q: u64, path: &str, rng_seed: u64) -> Self {
        let path_hash = Self::hash_path(path, rng_seed);
        let seed_lo = Self::wymix(
            Self::wymix(o ^ WYMIX_SALT_0, p ^ WYMIX_SALT_1),
            Self::wymix(q ^ WYMIX_SALT_2, path_hash ^ WYMIX_SALT_3),
        );
        let seed_hi = Self::wymix(
            Self::wymix(o ^ WYMIX_SALT_4, q ^ WYMIX_SALT_5),
            Self::wymix(p ^ WYMIX_SALT_6, path_hash ^ WYMIX_SALT_7),
        );
        PointGen { seed_lo, seed_hi }
    }

    /// Derive the i-th candidate point as (re, im) in [area.re_min, area.re_max) × [area.im_min, area.im_max).
    ///
    /// Stateless — any index can be computed independently of any other.
    #[inline]
    pub fn point(&self, index: u64, area: &Rect) -> (Fixed, Fixed) {
        let re_raw = Self::wymix(self.seed_lo ^ DERIVE_SALT_RE, index);
        let im_raw = Self::wymix(self.seed_hi ^ DERIVE_SALT_IM, index);

        let re_range = (area.re_max.0 as u64).wrapping_sub(area.re_min.0 as u64);
        let im_range = (area.im_max.0 as u64).wrapping_sub(area.im_min.0 as u64);

        let re = if re_range == 0 { area.re_min } else {
            Fixed(area.re_min.0.wrapping_add((re_raw % re_range) as i64))
        };
        let im = if im_range == 0 { area.im_min } else {
            Fixed(area.im_min.0.wrapping_add((im_raw % im_range) as i64))
        };
        (re, im)
    }
}

/// Right-shift applied to p0 to obtain the non-escaped point collision radius.
const NON_ESCAPED_RADIUS_SHIFT: u32 = 0;

/// Right-shift applied to p0 to obtain the low-escape point collision radius.
const LOW_ESCAPE_RADIUS_SHIFT: u32 = 0;
/// Uses usize::MAX to avoid colliding with real flood fill IDs (which count up from 0).
const NON_ESCAPED_FLOOD_ID: usize = usize::MAX;

/// Flood ID for low-escape-count points (below MIN_ESCAPE_COUNT_THRESHOLD).
const LOW_ESCAPE_FLOOD_ID: usize = usize::MAX - 1;

/// Stores flood-filled points as `(re, im, flood_id)` in a single
/// lexicographically ordered BTreeSet.  Efficient range queries on re
/// enable a single-pass collision test.
struct FloodFillStore {
    points: BTreeSet<(i64, i64, usize)>,
    next_id: usize,
}

impl FloodFillStore {
    fn new() -> Self {
        FloodFillStore {
            points: BTreeSet::new(),
            next_id: 0,
        }
    }

    /// Insert all points from a flood fill.  Returns the assigned flood ID.
    fn add_points(&mut self, pts: &[(Fixed, Fixed)]) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        for &(re, im) in pts {
            self.points.insert((re.0, im.0, id));
        }
        id
    }

    /// Insert a single non-escaping point (inside the set).
    fn add_non_escaped(&mut self, re: Fixed, im: Fixed) {
        self.points.insert((re.0, im.0, NON_ESCAPED_FLOOD_ID));
    }

    /// Insert a single low-escape-count point (below threshold).
    fn add_low_escape(&mut self, re: Fixed, im: Fixed) {
        self.points.insert((re.0, im.0, LOW_ESCAPE_FLOOD_ID));
    }

    /// Collision test: returns `true` when the query point is either
    ///   (a) an exact coincidence with any stored point, OR
    ///   (b) interior to a stored flood — i.e. a single flood ID has points
    ///       left, right, above AND below the query within radius.
    ///
    /// Uses `radius` for real flood IDs, `non_escaped_radius` for
    /// NON_ESCAPED_FLOOD_ID, and `low_escape_radius` for LOW_ESCAPE_FLOOD_ID.
    fn collides(&self, re: Fixed, im: Fixed, radius: i64, non_escaped_radius: i64, low_escape_radius: i64) -> bool {
        let qr = re.0;
        let qi = im.0;

        // Scan with the largest radius to capture all candidates.
        let scan_r = radius.max(non_escaped_radius).max(low_escape_radius);
        let lo = (qr.saturating_sub(scan_r), i64::MIN, 0usize);
        let hi = (qr.saturating_add(scan_r), i64::MAX, usize::MAX);

        // Per flood-ID direction bits: left=1 right=2 down=4 up=8.
        let mut seen: Vec<(usize, u8)> = Vec::new();

        for &(pr, pi, id) in self.points.range(lo..=hi) {
            // Use the appropriate radius for this point's ID.
            let r = match id {
                NON_ESCAPED_FLOOD_ID => non_escaped_radius,
                LOW_ESCAPE_FLOOD_ID => low_escape_radius,
                _ => radius,
            };
            if (pr - qr).abs() > r || (pi - qi).abs() > r {
                continue;
            }

            // Exact coincidence — immediate collision.
            if pr == qr && pi == qi {
                return true;
            }

            let mut bits: u8 = 0;
            if pr < qr { bits |= DIR_LEFT; }
            if pr > qr { bits |= DIR_RIGHT; }
            if pi < qi { bits |= DIR_DOWN; }
            if pi > qi { bits |= DIR_UP; }

            if let Some(entry) = seen.iter_mut().find(|(eid, _)| *eid == id) {
                entry.1 |= bits;
                if entry.1 & DIR_ALL == DIR_ALL {
                    return true;
                }
            } else {
                if bits & DIR_ALL == DIR_ALL {
                    return true; // single point at exact same re or im
                }
                seen.push((id, bits));
            }
        }
        false
    }
}

/// Result of a single flood fill operation.
enum FloodResult {
    /// Found a good island, plus all visited same-escape-count points.
    Good(Island, Vec<(Fixed, Fixed)>),
    /// Region too large at max pixel size.
    /// Carries (bbox, pixel_count, pixel_delta, visited points).
    TooLarge(Rect, u64, Fixed, Vec<(Fixed, Fixed)>),
}

/// Compute pixel delta from area's smaller side and shift.
/// pixel_delta = min(width, height) >> shift, always a power-of-two fraction
/// of the smaller side.  Used for collision radius computation.
///
/// The `side <= 0` guard defends against degenerate rectangles:
///  - **Zero side**: an empty rect (width or height is 0) would produce a
///    zero delta after shifting, but the early return avoids the redundant
///    arithmetic.
///  - **Negative side**: because `Fixed` wraps a signed `i64`, an inverted
///    rect (`re_min > re_max` or `im_min > im_max`) yields a negative
///    width/height.  Right-shifting a negative `i64` is an arithmetic shift
///    that preserves the sign bit, producing a large negative value —
///    nonsensical as a spatial step size.  The guard catches this and falls
///    back to the minimum representable delta `Fixed(1)`.
fn pixel_delta_from_shift(area: &Rect, shift: u32) -> Fixed {
    let side = std::cmp::min(area.width().0, area.height().0);
    if shift >= 63 || side <= 0 {
        return Fixed(1); // minimum
    }
    let d = side >> shift;
    if d <= 0 { Fixed(1) } else { Fixed(d) }
}

/// Compute initial pixel size p0 as the largest power of 2 such that
/// (width * height) / p0² >= min_grid_cells.
///
/// Equivalently, p0 = 2^floor(log2(sqrt(width * height / min_grid_cells))).
/// All arithmetic is integer (i128); no floating-point.
#[allow(unused_variables, unused_assignments)]
fn compute_p0(area: &Rect, min_grid_cells: u64) -> Fixed {
    let w = area.width().0 as i128;
    let h = area.height().0 as i128;
    if w <= 0 || h <= 0 || min_grid_cells == 0 {
        log_verbose!("[compute_p0] degenerate area: w={} h={} C={} → p0=1",
                  w, h, min_grid_cells);
        return Fixed(1);
    }
    let area_product = w * h; // i128, cannot overflow for I4F60 values
    let c = min_grid_cells as i128;
    let max_p0_sq = area_product / c;
    if max_p0_sq <= 0 {
        log_verbose!("[compute_p0] area/C underflow: area_product={} C={} → p0=1",
                  area_product, c);
        return Fixed(1);
    }
    // Largest power of 2 whose square ≤ max_p0_sq.
    // If max_p0_sq has B significant bits, then sqrt has ~B/2 bits,
    // so the answer is 2^floor((B-1)/2).
    let bits = 128 - (max_p0_sq as u128).leading_zeros(); // significant bits
    let p0_exp = (bits - 1) / 2;
    if p0_exp >= 63 {
        log_verbose!("[compute_p0] overflow guard: p0_exp={} → p0=1", p0_exp);
        return Fixed(1); // overflow guard
    }
    let p0 = 1i64 << p0_exp;
    let grid_cells = area_product / (p0 as i128 * p0 as i128);
    log_verbose!("[compute_p0] area={:.6}x{:.6} C={} → p0={} (2^{}) grid_cells={}",
              Fixed(w as i64).to_f64(), Fixed(h as i64).to_f64(),
              min_grid_cells, Fixed(p0).to_f64(), p0_exp, grid_cells);
    Fixed(p0)
}

/// Perform flood fill from a starting point.
/// Returns the result together with all visited same-escape-count points.
#[allow(unused_variables)]
fn flood_fill(
    start_re: Fixed,
    start_im: Fixed,
    target_esc: u32,
    area: &Rect,
    params: &DiscoveryParams,
    o: u64,
    p: u64,
    q: u64,
) -> FloodResult {
    let p0 = compute_p0(area, params.min_grid_cells);
    let delta = p0;

    log_verbose!("[flood_fill] start=({:.6},{:.6}) esc={} delta={:.6}",
              start_re.to_f64(), start_im.to_f64(), target_esc,
              delta.to_f64());

    let mut visited: HashSet<(i64, i64)> = HashSet::new();
    let mut queue: VecDeque<(Fixed, Fixed)> = VecDeque::new();
    let mut points: Vec<(Fixed, Fixed)> = Vec::new();
    queue.push_back((start_re, start_im));

    // Accumulate barycenter in i128 to avoid overflow
    let mut sum_re: i128 = 0;
    let mut sum_im: i128 = 0;
    let mut count: u64 = 0;
    let mut bbox = Rect {
        re_min: start_re,
        re_max: start_re,
        im_min: start_im,
        im_max: start_im,
    };

    // Track if we find a lower escape count point (closer to set boundary)
    let mut best_lower: Option<(Fixed, Fixed, u32)> = None;
    // Track if the flood fill propagated outside the current area
    let mut touched_boundary = false;

    while let Some((re, im)) = queue.pop_front() {
        if count >= params.max_flood_points {
            break;
        }

        // Discretize to grid for visited tracking
        let grid_re = re.0 / delta.0;
        let grid_im = im.0 / delta.0;
        if !visited.insert((grid_re, grid_im)) {
            continue;
        }

        if !area.contains(re, im) {
            touched_boundary = true;
            continue;
        }

        let esc = escape_count_generic(re, im, params.max_iter, o, p, q);

        match esc {
            None => continue, // inside set, boundary
            Some(e) if e > target_esc => {
                // Higher escape count — boundary (farther from set)
                continue;
            }
            Some(e) if e < target_esc => {
                // Lower escape count — closer to set boundary, track best
                if area.contains(re, im) {
                    match best_lower {
                        None => best_lower = Some((re, im, e)),
                        Some((_, _, prev_e)) if e < prev_e => {
                            best_lower = Some((re, im, e));
                        }
                        _ => {}
                    }
                }
                continue;
            }
            Some(_) => {
                // Same escape count — expand
                count += 1;
                sum_re += re.0 as i128;
                sum_im += im.0 as i128;
                points.push((re, im));

                // Expand bbox
                if re.0 < bbox.re_min.0 { bbox.re_min = re; }
                if re.0 > bbox.re_max.0 { bbox.re_max = re; }
                if im.0 < bbox.im_min.0 { bbox.im_min = im; }
                if im.0 > bbox.im_max.0 { bbox.im_max = im; }

                // Enqueue 4-neighbors
                if let Some(re_plus) = re.checked_add(delta) {
                    queue.push_back((re_plus, im));
                }
                if let Some(re_minus) = re.checked_sub(delta) {
                    queue.push_back((re_minus, im));
                }
                if let Some(im_plus) = im.checked_add(delta) {
                    queue.push_back((re, im_plus));
                }
                if let Some(im_minus) = im.checked_sub(delta) {
                    queue.push_back((re, im_minus));
                }
            }
        }
    }

    if count >= params.max_flood_points {
        log_verbose!("[flood_fill] → TooLarge: count={} delta={:.6}",
                  count, delta.to_f64());
        return FloodResult::TooLarge(bbox, count, delta, points);
    }

    // Flood fill halted naturally
    if count == 0 {
        // Isolated point — check if we found something lower nearby
        if let Some((l_re, l_im, l_esc)) = best_lower {
            if l_esc >= MIN_ESCAPE_COUNT_THRESHOLD {
                log_verbose!("[flood_fill] count=0, redirecting to lower esc={} at ({:.6},{:.6})",
                          l_esc, l_re.to_f64(), l_im.to_f64());
                return flood_fill(l_re, l_im, l_esc, area, params, o, p, q);
            } else {
                log_verbose!("[flood_fill] count=0, lower esc={} is sub-minimum (< {}), discarding as bad at ({:.6},{:.6})",
                          l_esc, MIN_ESCAPE_COUNT_THRESHOLD, start_re.to_f64(), start_im.to_f64());
            }
        } else {
            // Nothing found at all — treat as tiny exclusion (1 pixel at p0 resolution)
            log_verbose!("[flood_fill] → count=0, no neighbors, tiny exclusion at ({:.6},{:.6})",
                      start_re.to_f64(), start_im.to_f64());
        }
        let pt = vec![(start_re, start_im)];
        return FloodResult::TooLarge(Rect {
            re_min: start_re,
            re_max: start_re,
            im_min: start_im,
            im_max: start_im,
        }, 1, delta, pt);
    }

    // If we found a lower escape count, restart from there.
    // Lower escape means closer to the fractal boundary — more
    // structurally interesting and likely to form larger connected
    // regions (true islands).
    if let Some((l_re, l_im, l_esc)) = best_lower {
        if l_esc >= MIN_ESCAPE_COUNT_THRESHOLD {
            log_verbose!("[flood_fill] → Good but found lower esc={} (vs {}), redirecting",
                      l_esc, target_esc);
            return flood_fill(l_re, l_im, l_esc, area, params, o, p, q);
        } else {
            // Lower escape region is sub-minimum — the current region
            // borders noise near the fractal boundary.  Discard it as bad.
            log_verbose!("[flood_fill] → lower esc={} is sub-minimum (< {}), discarding region esc={} count={} as bad",
                      l_esc, MIN_ESCAPE_COUNT_THRESHOLD, target_esc, count);
            return FloodResult::TooLarge(bbox, count, delta, points);
        }
    }

    // If the flood fill reached outside the area, the island spans
    // the boundary and is not fully contained — treat as bad.
    if touched_boundary {
        log_verbose!("[flood_fill] → TooLarge (touched boundary): esc={} count={} delta={:.6}",
                  target_esc, count, delta.to_f64());
        return FloodResult::TooLarge(bbox, count, delta, points);
    }

    // Barycenter in Fixed: divide i128 sum by count
    let center_re = Fixed((sum_re / count as i128) as i64);
    let center_im = Fixed((sum_im / count as i128) as i64);

    log_verbose!("[flood_fill] → Good: esc={} count={} delta={:.6} center=({:.6},{:.6})",
              target_esc, count, delta.to_f64(),
              center_re.to_f64(), center_im.to_f64());

    let island = Island {
        center_re,
        center_im,
        pixel_count: count,
        pixel_delta: delta,
        escape_count: target_esc,
        bbox,
        score: 0, // computed after discovery
    };
    FloodResult::Good(island, points)
}

/// Compute the precise area contributed by a flood fill result.
/// Area = pixel_count * pixel_delta^2, computed in u128 to avoid overflow.
fn flood_area_u128(pixel_count: u64, pixel_delta: Fixed) -> u128 {
    let d = pixel_delta.0 as u128;
    pixel_count as u128 * d * d
}

/// Fixed-point log2 of a u128 value, with LOG2_FRAC_BITS fractional bits.
/// Returns 0 for inputs 0 or 1.
///
/// Algorithm: the integer part is the MSB position.  For the fractional part,
/// normalise v into a Q1.62 mantissa x ∈ [2^62, 2^63) and repeatedly square:
/// if x² ≥ 2·2^62 = 2^63, emit a 1-bit and halve; otherwise emit 0.
/// All arithmetic stays within u128 (62-bit mantissa squared fits in 124 bits).
fn fixed_log2_u128(v: u128) -> u128 {
    if v <= 1 {
        return 0;
    }
    let msb = 127 - v.leading_zeros(); // integer part of log2
    let int_part = (msb as u128) << LOG2_FRAC_BITS;

    // Normalise to Q1.62: shift v so the leading 1 lands at bit 62.
    // x represents the mantissa m = v/2^msb, with 1.0 = 2^62.
    let mut x: u128 = if msb <= 62 {
        v << (62 - msb)
    } else {
        v >> (msb - 62)
    };
    // x is in [2^62, 2^63).

    let mut frac: u128 = 0;
    for i in 1..=LOG2_FRAC_BITS {
        // x in [2^62, 2^63), so x*x in [2^124, 2^126) — fits in u128.
        // Divide by 2^62 to rescale: result in [2^62, 2^64).
        let sq = (x * x) >> 62;
        if sq >= (1u128 << 63) {
            // x² ≥ 2.0: emit 1, halve to stay in [2^62, 2^63)
            frac |= 1u128 << (LOG2_FRAC_BITS - i);
            x = sq >> 1;
        } else {
            x = sq;
        }
    }
    int_part | frac
}

/// Inherited seed point from a parent area's good island.
/// Each flood-filled point of a good island becomes a seed, so that
/// highly non-convex islands (whose barycenter may lie outside the
/// island) are still reliably re-discovered in the child area.
#[derive(Clone, Debug)]
pub struct InheritedSeed {
    pub re: Fixed,
    pub im: Fixed,
    pub escape_count: u32,
}

/// Discover islands within a rectangular area.
///
/// The PRNG is seeded from `path` (the bisection path string, e.g. "OlRd")
/// combined with `rng_seed`.
///
/// `inherited_seeds` are flood-filled points of good islands from the parent
/// area that fall within this child area.  They are re-flood-filled first
/// (before random sampling) to give the child a head start.
///
/// Returns `(islands, all_good_points)` where `all_good_points` contains
/// every flood-filled point (with escape count) from every good island,
/// suitable for filtering and passing as inherited seeds to child areas.
#[allow(unused_variables, unused_assignments)]
pub fn discover_islands(
    area: &Rect,
    params: &DiscoveryParams,
    rng_seed: u64,
    o: u64,
    p: u64,
    q: u64,
    path: &str,
    inherited_seeds: &[InheritedSeed],
) -> (Vec<Island>, Vec<InheritedSeed>) {
    let gen = PointGen::new(o, p, q, path, rng_seed);
    let mut islands: Vec<Island> = Vec::new();
    let mut store = FloodFillStore::new();

    // Collision radii:
    // - Flood-fill islands use the coarse radius (largest pixel size).
    // - Non-escaped points (inside the set) use a fine radius (p0 >> 3).
    // - Low-escape-count points use a slightly larger fine radius (p0 >> 2).
    let collision_radius = pixel_delta_from_shift(area, params.p_max_shift).0;
    let p0 = compute_p0(area, params.min_grid_cells).0;
    let non_escaped_radius = p0 >> NON_ESCAPED_RADIUS_SHIFT;
    let low_escape_radius = p0 >> LOW_ESCAPE_RADIUS_SHIFT;

    // Precise area tracking: sum of (pixel_count * pixel_delta^2) across all flood fills.
    let mut excluded_area: u128 = 0;
    // Total area of the search rectangle (u128 to avoid overflow).
    let total_area: u128 = area.width().0 as u128 * area.height().0 as u128;

    log_verbose!("[discover] area re=[{:.6},{:.6}] im=[{:.6},{:.6}] ({:.6}x{:.6}) \
              collision_r={:.6} total_area_raw={} inherited_seeds={}",
              area.re_min.to_f64(), area.re_max.to_f64(),
              area.im_min.to_f64(), area.im_max.to_f64(),
              area.width().to_f64(), area.height().to_f64(),
              Fixed(collision_radius).to_f64(), total_area,
              inherited_seeds.len());

    // Accumulates all flood-filled points from good islands (for inheritance).
    let mut all_good_points: Vec<InheritedSeed> = Vec::new();

    // Phase 1: Re-flood-fill inherited seeds from the parent area.
    // Many seeds will be redundant (same island at the new resolution);
    // the collision check skips them cheaply.
    let mut inherited_skip: u64 = 0;
    for (seed_idx, seed) in inherited_seeds.iter().enumerate() {
        // Skip seeds that already collide with a previously discovered region.
        if store.collides(seed.re, seed.im, collision_radius, non_escaped_radius, low_escape_radius) {
            inherited_skip += 1;
            continue;
        }
        let result = flood_fill(seed.re, seed.im, seed.escape_count, area, params, o, p, q);
        match result {
            FloodResult::Good(island, pts) => {
                excluded_area += flood_area_u128(island.pixel_count, island.pixel_delta);
                store.add_points(&pts);
                if island.pixel_count < MIN_ISLAND_PIXEL_COUNT {
                    log_verbose!("[discover] inherited seed {} → noise: esc={} pixels={} (< {}) → rejected",
                              seed_idx, island.escape_count, island.pixel_count,
                              MIN_ISLAND_PIXEL_COUNT);
                } else {
                    log_verbose!("[discover] inherited seed {} → island #{}: esc={} pixels={} delta={:.6} \
                              center=({:.6},{:.6})",
                              seed_idx, islands.len(),
                              island.escape_count, island.pixel_count,
                              island.pixel_delta.to_f64(),
                              island.center_re.to_f64(), island.center_im.to_f64());
                    // Collect all flood-filled points for inheritance.
                    for &(re, im) in &pts {
                        all_good_points.push(InheritedSeed { re, im, escape_count: island.escape_count });
                    }
                    islands.push(island);
                }
            }
            FloodResult::TooLarge(_bbox, pixel_count, pixel_delta, pts) => {
                excluded_area += flood_area_u128(pixel_count, pixel_delta);
                store.add_points(&pts);
                log_verbose!("[discover] inherited seed {} → TooLarge: pixels={} delta={:.6}",
                          seed_idx, pixel_count, pixel_delta.to_f64());
            }
        }
    }
    if inherited_skip > 0 {
        log_verbose!("[discover] inherited seeds: {} skipped by collision out of {}",
                  inherited_skip, inherited_seeds.len());
    }

    // Phase 2: Random sampling (same as before).
    let mut attempts: u64 = 0;
    let mut skip_collision: u64 = 0;
    let mut skip_non_esc: u64 = 0;
    let mut skip_low_esc: u64 = 0;

    while (islands.len() as u32) < params.target_good
        && excluded_area * EXCLUSION_DENOM < total_area * params.exclusion_threshold_num as u128
        && attempts < params.max_attempts
    {
        attempts += 1;

        let (re, im) = gen.point(attempts, area);

        // Single collision check handles flood-fill islands (coarse radius),
        // non-escaped points, and low-escape points in one scan.
        if store.collides(re, im, collision_radius, non_escaped_radius, low_escape_radius) {
            skip_collision += 1;
            continue;
        }

        let esc = escape_count_generic(re, im, params.max_iter, o, p, q);
        if esc.is_none() {
            skip_non_esc += 1;
            store.add_non_escaped(re, im);
            continue;
        }
        let esc = esc.unwrap();

        // Skip low escape counts — empirically, the first meaningful
        // islands only appear around escape count ~17.
        if esc < MIN_ESCAPE_COUNT_THRESHOLD {
            skip_low_esc += 1;
            store.add_low_escape(re, im);
            continue;
        }

        let result = flood_fill(re, im, esc, area, params, o, p, q);

        match result {
            FloodResult::Good(island, pts) => {
                excluded_area += flood_area_u128(island.pixel_count, island.pixel_delta);
                store.add_points(&pts);
                if island.pixel_count < MIN_ISLAND_PIXEL_COUNT {
                    // Too few pixels — noise, not a structural island.
                    // Points are already in the collision store.
                    log_verbose!("[discover] attempt {} → noise: esc={} pixels={} (< {}) → rejected",
                              attempts, island.escape_count, island.pixel_count,
                              MIN_ISLAND_PIXEL_COUNT);
                } else {
                    let bbox_w = (island.bbox.re_max.0 - island.bbox.re_min.0) as f64 / (1u64 << 60) as f64;
                    let bbox_h = (island.bbox.im_max.0 - island.bbox.im_min.0) as f64 / (1u64 << 60) as f64;
                    log_verbose!("[discover] attempt {} → island #{}: esc={} pixels={} delta={:.6} \
                              center=({:.6},{:.6}) bbox={:.6}x{:.6}",
                              attempts, islands.len(),
                              island.escape_count, island.pixel_count,
                              island.pixel_delta.to_f64(),
                              island.center_re.to_f64(), island.center_im.to_f64(),
                              bbox_w, bbox_h);
                    // Collect all flood-filled points for inheritance.
                    for &(re, im) in &pts {
                        all_good_points.push(InheritedSeed { re, im, escape_count: island.escape_count });
                    }
                    islands.push(island);
                }
            }
            FloodResult::TooLarge(_bbox, pixel_count, pixel_delta, pts) => {
                excluded_area += flood_area_u128(pixel_count, pixel_delta);
                store.add_points(&pts);
                log_verbose!("[discover] attempt {} → TooLarge: pixels={} delta={:.6}",
                          attempts, pixel_count, pixel_delta.to_f64());
            }
        }
    }

    // Compute score = log2(good_total_area / flood_area) as a fixed-point
    // u128 with LOG2_FRAC_BITS fractional bits.
    // good_total_area = sum of flood areas across Good islands only.
    // flood_area_j = pixel_count_j * pixel_delta_j^2.
    // All arithmetic is u128 — no floating-point.
    let good_total_area: u128 = islands.iter()
        .map(|i| flood_area_u128(i.pixel_count, i.pixel_delta))
        .sum();
    for island in &mut islands {
        let flood_area = flood_area_u128(island.pixel_count, island.pixel_delta);
        island.score = if flood_area > 0 && good_total_area > 0 {
            let ratio = good_total_area / flood_area;
            fixed_log2_u128(ratio)
        } else {
            MAX_DEGENERATE_SCORE
        };
    }

    // Stop-reason diagnosis
    let stop_reason = if (islands.len() as u32) >= params.target_good {
        "target_good"
    } else if excluded_area * EXCLUSION_DENOM >= total_area * params.exclusion_threshold_num as u128 {
        "exclusion"
    } else {
        "max_attempts"
    };

    let excl_pct = if total_area > 0 {
        (excluded_area as f64 / total_area as f64) * 100.0
    } else { 0.0 };

    // Per-island summary: pixel counts and escape counts
    let single_pixel_count = islands.iter().filter(|i| i.pixel_count == 1).count();
    let min_pixels = islands.iter().map(|i| i.pixel_count).min().unwrap_or(0);
    let max_pixels = islands.iter().map(|i| i.pixel_count).max().unwrap_or(0);
    let min_esc = islands.iter().map(|i| i.escape_count).min().unwrap_or(0);
    let max_esc = islands.iter().map(|i| i.escape_count).max().unwrap_or(0);

    log_verbose!("[discover] done: {} islands, {} attempts, stop={} \
              (skip: {} collision, {} non-esc, {} low-esc) excluded={:.1}% \
              pixels=[{},{}] single_px={} esc=[{},{}] good_area_raw={} good_pts={}",
              islands.len(), attempts, stop_reason,
              skip_collision, skip_non_esc, skip_low_esc, excl_pct,
              min_pixels, max_pixels, single_pixel_count,
              min_esc, max_esc, good_total_area, all_good_points.len());

    (islands, all_good_points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_log2_exact_powers_of_two() {
        // log2(2^k) should be exactly k << FRAC_BITS
        for k in 0..=64u32 {
            let v = 1u128 << k;
            let expected = (k as u128) << LOG2_FRAC_BITS;
            assert_eq!(fixed_log2_u128(v), expected, "log2(2^{k})");
        }
    }

    #[test]
    fn test_fixed_log2_zero_and_one() {
        assert_eq!(fixed_log2_u128(0), 0);
        assert_eq!(fixed_log2_u128(1), 0);
    }

    #[test]
    fn test_fixed_log2_non_powers() {
        // log2(3) ≈ 1.585; with 16 frac bits that's ~103,890
        let result = fixed_log2_u128(3);
        let expected_f64 = 3.0_f64.log2() * (1u64 << LOG2_FRAC_BITS) as f64;
        let diff = (result as f64 - expected_f64).abs();
        assert!(diff < 2.0, "log2(3): got {result}, expected ~{expected_f64:.0}, diff={diff}");

        // log2(1000) ≈ 9.9658
        let result = fixed_log2_u128(1000);
        let expected_f64 = 1000.0_f64.log2() * (1u64 << LOG2_FRAC_BITS) as f64;
        let diff = (result as f64 - expected_f64).abs();
        assert!(diff < 2.0, "log2(1000): got {result}, expected ~{expected_f64:.0}, diff={diff}");
    }

    #[test]
    fn test_fixed_log2_monotonic() {
        let mut prev = 0u128;
        for v in 2..=256u128 {
            let cur = fixed_log2_u128(v);
            assert!(cur >= prev, "log2 not monotonic at {v}: {cur} < {prev}");
            prev = cur;
        }
    }

    #[test]
    fn test_fixed_log2_large_values() {
        // log2(2^100 + 1) should be very close to 100 << 16
        let v = (1u128 << 100) + 1;
        let result = fixed_log2_u128(v);
        let expected = 100u128 << LOG2_FRAC_BITS;
        assert!((result as i128 - expected as i128).unsigned_abs() < 2,
                "log2(2^100+1): got {result}, expected ~{expected}");
    }
}
