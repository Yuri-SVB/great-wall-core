/// C FFI interface for the Python bridge.
///
/// Exposes encode, decode, and escape_count functions.

use crate::bisect::{encode, encode_with_seeds, decode, decode_full, EncodeResult};
use crate::discovery::InheritedSeed;
use crate::discovery::DiscoveryParams;
use crate::fixed::{Fixed, Rect, FRAC_BITS, INT_BITS};
use crate::fractal::{escape_count, escape_count_generic};
use crate::render_cache::RENDER_CACHE;
use crate::ENGINE_VERSION;
use rayon::prelude::*;
use std::slice;

/// Return the engine algorithm version string.
/// Writes up to `buf_len - 1` bytes + NUL terminator into `out_buf`.
/// Returns the version string length (excluding NUL).
#[no_mangle]
pub extern "C" fn bs_engine_version(out_buf: *mut u8, buf_len: u32) -> u32 {
    let v = ENGINE_VERSION.as_bytes();
    let copy_len = v.len().min(buf_len.saturating_sub(1) as usize);
    if !out_buf.is_null() && buf_len > 0 {
        let buf = unsafe { slice::from_raw_parts_mut(out_buf, buf_len as usize) };
        buf[..copy_len].copy_from_slice(&v[..copy_len]);
        buf[copy_len] = 0; // NUL
    }
    v.len() as u32
}

/// Maximum escape count value for u8 pixel output (0 reserved for non-escaping).
/// Escaped pixels are mapped to 1..=255 via (count % 255) + 1.
const PIXEL_ESCAPE_MOD: u32 = 255;

/// Default maximum discovery attempts (for FFI callers that don't expose this param).
pub(crate) const DEFAULT_MAX_ATTEMPTS: u64 = 500_000;

/// Compute escape count for a single point.
/// Returns the iteration count, or -1 if the point does not escape.
#[no_mangle]
pub extern "C" fn bs_escape_count(c_re: f64, c_im: f64, max_iter: u32) -> i32 {
    let re = Fixed::from_f64(c_re);
    let im = Fixed::from_f64(c_im);
    match escape_count(re, im, max_iter) {
        Some(i) => i as i32,
        None => -1,
    }
}

/// Render an escape-count map for a viewport.
///
/// Output: out_pixels[y * out_w + x] = escape count (clamped to 255), or 0 for non-escaping.
///
/// If a render cache has been initialized (via `bs_cache_init`), cached results
/// are reused and new results are inserted.  The cache is keyed on raw Fixed
/// coordinates, so panning at the same zoom level gets near-perfect reuse.
#[no_mangle]
pub extern "C" fn bs_render_viewport(
    origin_re: f64,
    origin_im: f64,
    step: f64,
    out_w: i32,
    out_h: i32,
    max_iter: u32,
    out_pixels: *mut u8,
) {
    let pixels = unsafe { slice::from_raw_parts_mut(out_pixels, (out_w * out_h) as usize) };

    // Pre-compute Fixed coordinates for all pixels.
    // This lets us do a single cache-read pass, compute misses in parallel,
    // then a single cache-write pass — minimising lock contention.
    let w = out_w as usize;
    let h = out_h as usize;

    // Build coordinate grid
    let mut coords: Vec<(Fixed, Fixed)> = Vec::with_capacity(w * h);
    for py in 0..h {
        let im = Fixed::from_f64(origin_im + py as f64 * step);
        for px in 0..w {
            let re = Fixed::from_f64(origin_re + px as f64 * step);
            coords.push((re, im));
        }
    }

    // Try cache read pass
    let cache_guard = RENDER_CACHE.lock().unwrap();
    let cache_active = cache_guard.is_some();

    if cache_active {
        let cache = cache_guard.as_ref().unwrap();
        // Mark hits directly into pixels; collect miss indices
        let mut miss_indices: Vec<usize> = Vec::new();
        for (i, &(re, im)) in coords.iter().enumerate() {
            if let Some(&esc) = cache.get(&(re.0, im.0)) {
                pixels[i] = match esc {
                    Some(e) => ((e % PIXEL_ESCAPE_MOD) + 1) as u8,
                    None => 0,
                };
            } else {
                miss_indices.push(i);
            }
        }
        drop(cache_guard);

        // Compute misses in parallel
        let results: Vec<(usize, Option<u32>)> = miss_indices
            .par_iter()
            .map(|&i| {
                let (re, im) = coords[i];
                (i, escape_count(re, im, max_iter))
            })
            .collect();

        // Write results to pixels and cache
        let mut cache_guard = RENDER_CACHE.lock().unwrap();
        let cache = cache_guard.as_mut().unwrap();
        for (i, esc) in results {
            let (re, im) = coords[i];
            pixels[i] = match esc {
                Some(e) => ((e % PIXEL_ESCAPE_MOD) + 1) as u8,
                None => 0,
            };
            cache.insert((re.0, im.0), esc);
        }
    } else {
        drop(cache_guard);

        // No cache — original parallel-by-row path
        pixels
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(py, row)| {
                let c_im = origin_im + py as f64 * step;
                let im = Fixed::from_f64(c_im);
                for (px, pixel) in row.iter_mut().enumerate() {
                    let c_re = origin_re + px as f64 * step;
                    let re = Fixed::from_f64(c_re);
                    *pixel = match escape_count(re, im, max_iter) {
                        Some(i) => ((i % 255) + 1) as u8,
                        None => 0,
                    };
                }
            });
    }
}

/// Render a viewport using the perturbed Burning Ship formula:
///
///   z_{n+1} = (|Re(z_n)| + Re(p) + i·(|Im(z_n)| + Im(p)))² + c
///
/// The 64-bit parameter `p` encodes the additive perturbation via `decode_perturbation_p`.
/// Uses a separate stage-2 render cache (RENDER_CACHE_STAGE2).
#[no_mangle]
pub extern "C" fn bs_render_viewport_generic(
    origin_re: f64,
    origin_im: f64,
    step: f64,
    out_w: i32,
    out_h: i32,
    max_iter: u32,
    o: u64,
    p: u64,
    q: u64,
    out_pixels: *mut u8,
) {
    let pixels = unsafe { slice::from_raw_parts_mut(out_pixels, (out_w * out_h) as usize) };
    let w = out_w as usize;
    let h = out_h as usize;

    // Build coordinate grid
    let mut coords: Vec<(Fixed, Fixed)> = Vec::with_capacity(w * h);
    for py in 0..h {
        let im = Fixed::from_f64(origin_im + py as f64 * step);
        for px in 0..w {
            let re = Fixed::from_f64(origin_re + px as f64 * step);
            coords.push((re, im));
        }
    }

    // Try stage-2 cache
    use crate::render_cache::RENDER_CACHE_STAGE2;
    let cache_guard = RENDER_CACHE_STAGE2.lock().unwrap();
    let cache_active = cache_guard.is_some();

    if cache_active {
        let cache = cache_guard.as_ref().unwrap();
        let mut miss_indices: Vec<usize> = Vec::new();
        for (i, &(re, im)) in coords.iter().enumerate() {
            if let Some(&esc) = cache.get(&(re.0, im.0)) {
                pixels[i] = match esc {
                    Some(e) => ((e % PIXEL_ESCAPE_MOD) + 1) as u8,
                    None => 0,
                };
            } else {
                miss_indices.push(i);
            }
        }
        drop(cache_guard);

        let results: Vec<(usize, Option<u32>)> = miss_indices
            .par_iter()
            .map(|&i| {
                let (re, im) = coords[i];
                (i, escape_count_generic(re, im, max_iter, o, p, q))
            })
            .collect();

        let mut cache_guard = RENDER_CACHE_STAGE2.lock().unwrap();
        let cache = cache_guard.as_mut().unwrap();
        for (i, esc) in results {
            let (re, im) = coords[i];
            pixels[i] = match esc {
                Some(e) => ((e % PIXEL_ESCAPE_MOD) + 1) as u8,
                None => 0,
            };
            cache.insert((re.0, im.0), esc);
        }
    } else {
        drop(cache_guard);
        pixels
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(py, row)| {
                let im = Fixed::from_f64(origin_im + py as f64 * step);
                for (px, pixel) in row.iter_mut().enumerate() {
                    let re = Fixed::from_f64(origin_re + px as f64 * step);
                    *pixel = match escape_count_generic(re, im, max_iter, o, p, q) {
                        Some(i) => ((i % 255) + 1) as u8,
                        None => 0,
                    };
                }
            });
    }
}

/// Opaque handle for encode results.
pub struct EncodeResultHandle {
    result: EncodeResult,
}

/// Encode a bit array into a fractal location.
///
/// Returns an opaque handle. Use bs_encode_result_* functions to query it.
/// Caller must free with bs_encode_result_free.
///
/// Area bounds are Fixed-point values (repr(transparent) over i64).
#[no_mangle]
pub extern "C" fn bs_encode(
    seed_bits: *const u8,
    num_bits: u32,
    // Initial area (Fixed-point)
    area_re_min: Fixed,
    area_re_max: Fixed,
    area_im_min: Fixed,
    area_im_max: Fixed,
    // Discovery params
    max_iter: u32,
    target_good: u32,
    max_flood_points: u64,
    min_grid_cells: u64,
    p_max_shift: u32,
    exclusion_threshold_num: u32,
    // RNG seed
    rng_seed: u64,
    // Perturbation parameters (orbit seed o, additive shift p, linear term q)
    o: u64,
    p: u64,
    q: u64,
    // Path prefix (accumulated from prior points; NUL-terminated UTF-8)
    path_prefix: *const u8,
    path_prefix_len: u32,
) -> *mut EncodeResultHandle {
    let bits = unsafe { slice::from_raw_parts(seed_bits, num_bits as usize) };
    let prefix = if path_prefix.is_null() || path_prefix_len == 0 {
        "O"
    } else {
        unsafe { std::str::from_utf8_unchecked(slice::from_raw_parts(path_prefix, path_prefix_len as usize)) }
    };
    let area = Rect { re_min: area_re_min, re_max: area_re_max, im_min: area_im_min, im_max: area_im_max };
    let params = DiscoveryParams {
        max_iter,
        min_grid_cells,
        p_max_shift,
        max_flood_points,
        target_good,
        exclusion_threshold_num,
        max_attempts: DEFAULT_MAX_ATTEMPTS,
    };

    let result = encode(bits, area, &params, rng_seed, o, p, q, prefix);
    let handle = Box::new(EncodeResultHandle { result });
    Box::into_raw(handle)
}

/// Get the encoded point as Fixed values.
#[no_mangle]
pub extern "C" fn bs_encode_result_point(handle: *const EncodeResultHandle, out_re: *mut Fixed, out_im: *mut Fixed) {
    let h = unsafe { &*handle };
    unsafe {
        *out_re = h.result.point_re;
        *out_im = h.result.point_im;
    }
}

/// Get the number of bits consumed.
#[no_mangle]
pub extern "C" fn bs_encode_result_bits(handle: *const EncodeResultHandle) -> u32 {
    let h = unsafe { &*handle };
    h.result.bits_consumed as u32
}

/// Get the number of bisection steps.
#[no_mangle]
pub extern "C" fn bs_encode_result_num_steps(handle: *const EncodeResultHandle) -> u32 {
    let h = unsafe { &*handle };
    h.result.steps.len() as u32
}

/// Get the bisection path string (e.g. "OlRdLu").
///
/// Writes up to `buf_len` bytes (including NUL terminator) into `out_buf`.
/// Returns the total length of the path (excluding NUL), even if truncated.
#[no_mangle]
pub extern "C" fn bs_encode_result_path(
    handle: *const EncodeResultHandle,
    out_buf: *mut u8,
    buf_len: u32,
) -> u32 {
    let h = unsafe { &*handle };
    let path_bytes = h.result.path.as_bytes();
    let path_len = path_bytes.len();
    if !out_buf.is_null() && buf_len > 0 {
        let copy_len = std::cmp::min(path_len, (buf_len - 1) as usize);
        let out = unsafe { slice::from_raw_parts_mut(out_buf, buf_len as usize) };
        out[..copy_len].copy_from_slice(&path_bytes[..copy_len]);
        out[copy_len] = 0; // NUL terminator
    }
    path_len as u32
}

/// Get a bisection step's data.
///
/// split_coord is a Fixed-point value.
/// Contraction is returned as separate numerator and denominator (u64).
#[no_mangle]
pub extern "C" fn bs_encode_result_step(
    handle: *const EncodeResultHandle,
    step_idx: u32,
    out_split_coord: *mut Fixed,
    out_split_vertical: *mut i32,
    out_bit: *mut u8,
    out_chose_larger: *mut i32,
    out_contraction_num: *mut u64,
    out_contraction_den: *mut u64,
    out_islands_found: *mut u32,
) {
    let h = unsafe { &*handle };
    let step = &h.result.steps[step_idx as usize];
    unsafe {
        *out_split_coord = step.split_coord;
        *out_split_vertical = step.split_vertical as i32;
        *out_bit = step.bit;
        *out_chose_larger = step.chose_larger as i32;
        *out_contraction_num = step.contraction_num;
        *out_contraction_den = step.contraction_den;
        *out_islands_found = step.islands_found as u32;
    }
}

/// Get the area rectangle for a bisection step (the rect BEFORE that split).
///
/// All bounds are Fixed-point values.
#[no_mangle]
pub extern "C" fn bs_encode_result_step_area(
    handle: *const EncodeResultHandle,
    step_idx: u32,
    out_re_min: *mut Fixed,
    out_re_max: *mut Fixed,
    out_im_min: *mut Fixed,
    out_im_max: *mut Fixed,
) {
    let h = unsafe { &*handle };
    let step = &h.result.steps[step_idx as usize];
    unsafe {
        *out_re_min = step.area.re_min;
        *out_re_max = step.area.re_max;
        *out_im_min = step.area.im_min;
        *out_im_max = step.area.im_max;
    }
}

/// Get the number of good islands discovered at a bisection step.
#[no_mangle]
pub extern "C" fn bs_encode_result_step_num_islands(
    handle: *const EncodeResultHandle,
    step_idx: u32,
) -> u32 {
    let h = unsafe { &*handle };
    let step = &h.result.steps[step_idx as usize];
    step.island_centers.len() as u32
}

/// Get an island's barycenter and pixel count for a bisection step.
///
/// island_idx must be < bs_encode_result_step_num_islands().
#[no_mangle]
pub extern "C" fn bs_encode_result_step_island(
    handle: *const EncodeResultHandle,
    step_idx: u32,
    island_idx: u32,
    out_re: *mut Fixed,
    out_im: *mut Fixed,
    out_pixel_count: *mut u64,
) {
    let h = unsafe { &*handle };
    let step = &h.result.steps[step_idx as usize];
    let (re, im, px) = step.island_centers[island_idx as usize];
    unsafe {
        *out_re = re;
        *out_im = im;
        *out_pixel_count = px;
    }
}

/// Get the final rectangle as Fixed-point values.
#[no_mangle]
pub extern "C" fn bs_encode_result_final_rect(
    handle: *const EncodeResultHandle,
    out_re_min: *mut Fixed,
    out_re_max: *mut Fixed,
    out_im_min: *mut Fixed,
    out_im_max: *mut Fixed,
) {
    let h = unsafe { &*handle };
    unsafe {
        *out_re_min = h.result.final_rect.re_min;
        *out_re_max = h.result.final_rect.re_max;
        *out_im_min = h.result.final_rect.im_min;
        *out_im_max = h.result.final_rect.im_max;
    }
}

/// Get the number of inherited seeds in the encode result.
#[no_mangle]
pub extern "C" fn bs_encode_result_num_seeds(handle: *const EncodeResultHandle) -> u32 {
    let h = unsafe { &*handle };
    h.result.inherited_seeds.len() as u32
}

/// Copy inherited seeds to caller buffer.
/// Each seed is written as (re: i64, im: i64, escape_count: u32, _pad: u32) = 24 bytes.
/// Buffer must hold at least num_seeds * 24 bytes.
#[no_mangle]
pub extern "C" fn bs_encode_result_seeds(
    handle: *const EncodeResultHandle,
    out_buf: *mut u8,
    buf_len: u32,
) {
    let h = unsafe { &*handle };
    let seeds = &h.result.inherited_seeds;
    let needed = seeds.len() * 24;
    assert!(buf_len as usize >= needed);
    let buf = unsafe { slice::from_raw_parts_mut(out_buf, buf_len as usize) };
    for (i, seed) in seeds.iter().enumerate() {
        let off = i * 24;
        buf[off..off+8].copy_from_slice(&seed.re.0.to_le_bytes());
        buf[off+8..off+16].copy_from_slice(&seed.im.0.to_le_bytes());
        buf[off+16..off+20].copy_from_slice(&seed.escape_count.to_le_bytes());
        buf[off+20..off+24].copy_from_slice(&0u32.to_le_bytes()); // padding
    }
}

/// Encode with initial inherited seeds (for incremental encoding).
/// seeds_buf contains num_seeds records of 24 bytes each: (re: i64 LE, im: i64 LE, esc: u32 LE, pad: u32).
#[no_mangle]
pub extern "C" fn bs_encode_with_seeds(
    seed_bits: *const u8,
    num_bits: u32,
    area_re_min: Fixed,
    area_re_max: Fixed,
    area_im_min: Fixed,
    area_im_max: Fixed,
    max_iter: u32,
    target_good: u32,
    max_flood_points: u64,
    min_grid_cells: u64,
    p_max_shift: u32,
    exclusion_threshold_num: u32,
    rng_seed: u64,
    o: u64,
    p: u64,
    q: u64,
    path_prefix: *const u8,
    path_prefix_len: u32,
    seeds_buf: *const u8,
    num_seeds: u32,
) -> *mut EncodeResultHandle {
    let bits = unsafe { slice::from_raw_parts(seed_bits, num_bits as usize) };
    let prefix = if path_prefix.is_null() || path_prefix_len == 0 {
        "O"
    } else {
        unsafe { std::str::from_utf8_unchecked(slice::from_raw_parts(path_prefix, path_prefix_len as usize)) }
    };
    let area = Rect { re_min: area_re_min, re_max: area_re_max, im_min: area_im_min, im_max: area_im_max };
    let params = DiscoveryParams {
        max_iter,
        min_grid_cells,
        p_max_shift,
        max_flood_points,
        target_good,
        exclusion_threshold_num,
        max_attempts: DEFAULT_MAX_ATTEMPTS,
    };

    // Deserialize inherited seeds
    let mut initial_seeds = Vec::with_capacity(num_seeds as usize);
    if num_seeds > 0 && !seeds_buf.is_null() {
        let buf = unsafe { slice::from_raw_parts(seeds_buf, num_seeds as usize * 24) };
        for i in 0..num_seeds as usize {
            let off = i * 24;
            let re = i64::from_le_bytes(buf[off..off+8].try_into().unwrap());
            let im = i64::from_le_bytes(buf[off+8..off+16].try_into().unwrap());
            let esc = u32::from_le_bytes(buf[off+16..off+20].try_into().unwrap());
            initial_seeds.push(InheritedSeed { re: Fixed(re), im: Fixed(im), escape_count: esc });
        }
    }

    let result = encode_with_seeds(bits, area, &params, rng_seed, o, p, q, prefix, initial_seeds);
    let handle = Box::new(EncodeResultHandle { result });
    Box::into_raw(handle)
}

/// Free an encode result handle.
#[no_mangle]
pub extern "C" fn bs_encode_result_free(handle: *mut EncodeResultHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Decode a point back to bits.
///
/// All coordinates are Fixed-point values for exact reproducibility.
/// Writes the decoded bits to out_bits (must be pre-allocated with num_bits bytes).
#[no_mangle]
pub extern "C" fn bs_decode(
    point_re: Fixed,
    point_im: Fixed,
    num_bits: u32,
    area_re_min: Fixed,
    area_re_max: Fixed,
    area_im_min: Fixed,
    area_im_max: Fixed,
    max_iter: u32,
    target_good: u32,
    max_flood_points: u64,
    min_grid_cells: u64,
    p_max_shift: u32,
    exclusion_threshold_num: u32,
    rng_seed: u64,
    o: u64,
    p: u64,
    q: u64,
    path_prefix: *const u8,
    path_prefix_len: u32,
    out_bits: *mut u8,
) {
    let prefix = if path_prefix.is_null() || path_prefix_len == 0 {
        "O"
    } else {
        unsafe { std::str::from_utf8_unchecked(slice::from_raw_parts(path_prefix, path_prefix_len as usize)) }
    };
    let area = Rect { re_min: area_re_min, re_max: area_re_max, im_min: area_im_min, im_max: area_im_max };
    let params = DiscoveryParams {
        max_iter,
        min_grid_cells,
        p_max_shift,
        max_flood_points,
        target_good,
        exclusion_threshold_num,
        max_attempts: DEFAULT_MAX_ATTEMPTS,
    };

    let bits = decode(point_re, point_im, num_bits as usize, area, &params, rng_seed, o, p, q, prefix);
    let out = unsafe { slice::from_raw_parts_mut(out_bits, num_bits as usize) };
    out.copy_from_slice(&bits);
}

/// Decode a point back to bits, also returning the final leaf rectangle and
/// a validity flag (1 = valid, 0 = point fell in a contracted-away region).
///
/// All coordinates are Fixed-point values.
/// out_bits must be pre-allocated with num_bits bytes.
/// out_rect receives [re_min, re_max, im_min, im_max] as Fixed.
#[no_mangle]
pub extern "C" fn bs_decode_full(
    point_re: Fixed,
    point_im: Fixed,
    num_bits: u32,
    area_re_min: Fixed,
    area_re_max: Fixed,
    area_im_min: Fixed,
    area_im_max: Fixed,
    max_iter: u32,
    target_good: u32,
    max_flood_points: u64,
    min_grid_cells: u64,
    p_max_shift: u32,
    exclusion_threshold_num: u32,
    rng_seed: u64,
    o: u64,
    p: u64,
    q: u64,
    path_prefix: *const u8,
    path_prefix_len: u32,
    out_bits: *mut u8,
    out_rect: *mut Fixed,
    out_valid: *mut u8,
    // Path output buffer (caller-provided, NUL-terminated).
    // out_path_len receives the actual path length (excluding NUL).
    out_path: *mut u8,
    out_path_buf_len: u32,
    out_path_len: *mut u32,
) {
    let prefix = if path_prefix.is_null() || path_prefix_len == 0 {
        "O"
    } else {
        unsafe { std::str::from_utf8_unchecked(slice::from_raw_parts(path_prefix, path_prefix_len as usize)) }
    };
    let area = Rect { re_min: area_re_min, re_max: area_re_max, im_min: area_im_min, im_max: area_im_max };
    let params = DiscoveryParams {
        max_iter,
        min_grid_cells,
        p_max_shift,
        max_flood_points,
        target_good,
        exclusion_threshold_num,
        max_attempts: DEFAULT_MAX_ATTEMPTS,
    };

    let (bits, final_rect, valid, path, _steps) = decode_full(
        point_re, point_im,
        num_bits as usize, area, &params, rng_seed, o, p, q, prefix,
    );
    let out = unsafe { slice::from_raw_parts_mut(out_bits, num_bits as usize) };
    out.copy_from_slice(&bits);
    unsafe {
        let rect = slice::from_raw_parts_mut(out_rect, 4);
        rect[0] = final_rect.re_min;
        rect[1] = final_rect.re_max;
        rect[2] = final_rect.im_min;
        rect[3] = final_rect.im_max;
        *out_valid = if valid { 1 } else { 0 };
        // Write path
        let path_bytes = path.as_bytes();
        if !out_path_len.is_null() {
            *out_path_len = path_bytes.len() as u32;
        }
        if !out_path.is_null() && out_path_buf_len > 0 {
            let copy_len = std::cmp::min(path_bytes.len(), (out_path_buf_len - 1) as usize);
            let pbuf = slice::from_raw_parts_mut(out_path, out_path_buf_len as usize);
            pbuf[..copy_len].copy_from_slice(&path_bytes[..copy_len]);
            pbuf[copy_len] = 0;
        }
    }
}

/// Query the current fixed-point configuration.
#[no_mangle]
pub extern "C" fn bs_get_precision(out_frac_bits: *mut u32, out_int_bits: *mut u32) {
    unsafe {
        *out_frac_bits = FRAC_BITS;
        *out_int_bits = INT_BITS;
    }
}

// -----------------------------------------------------------------------
// Render cache management
// -----------------------------------------------------------------------

use crate::render_cache::FifoCache;

/// Initialize the global render cache with the given capacity (number of entries).
/// If a cache already exists, it is replaced.
#[no_mangle]
pub extern "C" fn bs_cache_init(capacity: u64) {
    let mut guard = RENDER_CACHE.lock().unwrap();
    *guard = Some(FifoCache::new(capacity as usize));
}

/// Clear all entries from the render cache without deallocating.
#[no_mangle]
pub extern "C" fn bs_cache_clear() {
    let mut guard = RENDER_CACHE.lock().unwrap();
    if let Some(cache) = guard.as_mut() {
        cache.clear();
    }
}

/// Destroy the render cache, freeing its memory.
#[no_mangle]
pub extern "C" fn bs_cache_destroy() {
    let mut guard = RENDER_CACHE.lock().unwrap();
    *guard = None;
}

/// Return the current number of entries in the cache (0 if not initialized).
#[no_mangle]
pub extern "C" fn bs_cache_len() -> u64 {
    let guard = RENDER_CACHE.lock().unwrap();
    match guard.as_ref() {
        Some(cache) => cache.len() as u64,
        None => 0,
    }
}

// -----------------------------------------------------------------------
// Stage-2 render cache management
// -----------------------------------------------------------------------

use crate::render_cache::RENDER_CACHE_STAGE2;

#[no_mangle]
pub extern "C" fn bs_cache_init_stage2(capacity: u64) {
    let mut guard = RENDER_CACHE_STAGE2.lock().unwrap();
    *guard = Some(FifoCache::new(capacity as usize));
}

#[no_mangle]
pub extern "C" fn bs_cache_clear_stage2() {
    let mut guard = RENDER_CACHE_STAGE2.lock().unwrap();
    if let Some(cache) = guard.as_mut() {
        cache.clear();
    }
}

#[no_mangle]
pub extern "C" fn bs_cache_destroy_stage2() {
    let mut guard = RENDER_CACHE_STAGE2.lock().unwrap();
    *guard = None;
}

// -----------------------------------------------------------------------
// Argon2 iterative hashing
// -----------------------------------------------------------------------

use crate::argon2_hash::{iterative_argon2, argon2_single, argon2id_master};

/// Master-secret export (protocol 0.3.0): one Argon2id pass over the
/// reproducible setup transcript.
///
/// - `msg_ptr` / `msg_len`: the transcript message bytes.
/// - `out_ptr` / `out_len`: caller-provided output buffer (the Argon2 output
///   length `l`; the protocol uses 1024 bytes).
///
/// Uses Argon2id with the fixed master profile (m = 64 MiB, t = 8, p = 2) and
/// the fixed salt `b"greatwall"`.
#[no_mangle]
pub extern "C" fn bs_argon2id_master(
    msg_ptr: *const u8,
    msg_len: u32,
    out_ptr: *mut u8,
    out_len: u32,
) {
    let message = unsafe { slice::from_raw_parts(msg_ptr, msg_len as usize) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, out_len as usize) };
    argon2id_master(message, out);
}

/// Run iterative Argon2d hashing on 8 bytes of stage-1 entropy.
///
/// - `input_ptr`: pointer to 8 bytes (64-bit stage-1 entropy)
/// - `gui_iterations`: number of hash-then-feed-back cycles (1 = single pass)
/// - `profile`: 0 = Basic (mobile), 1 = Advanced (desktop), 2 = Great Wall (server)
/// - `out_digest`: pointer to 32-byte output buffer for the final digest
#[no_mangle]
pub extern "C" fn bs_argon2_hash(
    input_ptr: *const u8,
    gui_iterations: u32,
    profile: u8,
    out_digest: *mut u8,
) {
    let input: [u8; 8] = unsafe {
        let slice = slice::from_raw_parts(input_ptr, 8);
        let mut arr = [0u8; 8];
        arr.copy_from_slice(slice);
        arr
    };
    let digest = iterative_argon2(&input, gui_iterations, profile);
    unsafe {
        let out = slice::from_raw_parts_mut(out_digest, 32);
        out.copy_from_slice(&digest);
    }
}

/// Run a single Argon2d pass on arbitrary-length input.
///
/// Designed to be called in a loop from Python for progress reporting.
/// - `input_ptr`: pointer to input bytes (8 bytes on first call, 32 on subsequent)
/// - `input_len`: length of input in bytes
/// - `profile`: 0 = Basic (mobile), 1 = Advanced (desktop), 2 = Great Wall (server)
/// - `out_digest`: pointer to 32-byte output buffer
#[no_mangle]
pub extern "C" fn bs_argon2_single(
    input_ptr: *const u8,
    input_len: u32,
    profile: u8,
    out_digest: *mut u8,
) {
    let input = unsafe { slice::from_raw_parts(input_ptr, input_len as usize) };
    let digest = argon2_single(input, profile);
    unsafe {
        let out = slice::from_raw_parts_mut(out_digest, 32);
        out.copy_from_slice(&digest);
    }
}

/// Canonicalize a stage-0 salt/pepper string to the protocol's `[A-Z0-9-]`
/// set (up-case ASCII letters, drop everything else). This is the single
/// source of truth for the rule (DESIGN.md "Strong text restrictions"); the
/// wallet calls it rather than reimplementing the filter, so the same text
/// yields byte-identical seeds across the engine and the GUI.
///
/// Query-then-fill: pass `out_ptr = null` / `out_cap = 0` to learn the
/// canonical length, then call again with a buffer of that size. Writes up to
/// `out_cap` ASCII bytes (no NUL terminator) and always returns the full
/// canonical length in bytes.
#[no_mangle]
pub extern "C" fn bs_salt_pepper_canonicalize(
    in_ptr: *const u8,
    in_len: u32,
    out_ptr: *mut u8,
    out_cap: u32,
) -> u32 {
    let input: &[u8] = if in_ptr.is_null() || in_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(in_ptr, in_len as usize) }
    };
    let canon = crate::text::canonicalize_stage_text(input);
    if !out_ptr.is_null() && out_cap > 0 {
        let n = canon.len().min(out_cap as usize);
        let out = unsafe { slice::from_raw_parts_mut(out_ptr, out_cap as usize) };
        out[..n].copy_from_slice(&canon[..n]);
    }
    canon.len() as u32
}

/// Build one chain link's Argon2 input: the canonical stage-0 text bytes
/// followed by `bits_to_bytes(prior_bits)` (the protocol byte layout from
/// `argon2_pipeline.derive_stage_params`). `bits_ptr` points at `n_bits`
/// 0/1 bytes — the concatenated bits of every preceding point.
///
/// Query-then-fill like [`bs_salt_pepper_canonicalize`]: pass
/// `out_ptr = null` / `out_cap = 0` to learn the length, then call again with
/// a buffer of that size. Always returns the full input length in bytes.
#[no_mangle]
pub extern "C" fn bs_chain_input(
    text_ptr: *const u8,
    text_len: u32,
    bits_ptr: *const u8,
    n_bits: u32,
    out_ptr: *mut u8,
    out_cap: u32,
) -> u32 {
    let text: &[u8] = if text_ptr.is_null() || text_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(text_ptr, text_len as usize) }
    };
    let bits: &[u8] = if bits_ptr.is_null() || n_bits == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(bits_ptr, n_bits as usize) }
    };
    let data = crate::text::chain_input(text, bits);
    if !out_ptr.is_null() && out_cap > 0 {
        let n = data.len().min(out_cap as usize);
        let out = unsafe { slice::from_raw_parts_mut(out_ptr, out_cap as usize) };
        out[..n].copy_from_slice(&data[..n]);
    }
    data.len() as u32
}

/// Fill the canonical encode/decode discovery parameters — the single source of
/// truth callers must use instead of declaring their own. The engine dictates
/// the protocol; a stale hand-mirrored copy of these (e.g. `max_iter = 64`) is
/// what stalled deep-zoom encodes. Pass the same values straight back into
/// `bs_encode` / `bs_decode_full`. Any out pointer may be null to skip it.
#[no_mangle]
pub extern "C" fn bs_encode_params(
    out_max_iter: *mut u32,
    out_target_good: *mut u32,
    out_max_flood_points: *mut u64,
    out_min_grid_cells: *mut u64,
    out_p_max_shift: *mut u32,
    out_exclusion_threshold_num: *mut u32,
    out_rng_seed: *mut u64,
) {
    use crate::protocol;
    unsafe {
        if !out_max_iter.is_null() {
            *out_max_iter = protocol::ENCODE_MAX_ITER;
        }
        if !out_target_good.is_null() {
            *out_target_good = protocol::ENCODE_TARGET_GOOD;
        }
        if !out_max_flood_points.is_null() {
            *out_max_flood_points = protocol::ENCODE_MAX_FLOOD_POINTS;
        }
        if !out_min_grid_cells.is_null() {
            *out_min_grid_cells = protocol::ENCODE_MIN_GRID_CELLS;
        }
        if !out_p_max_shift.is_null() {
            *out_p_max_shift = protocol::ENCODE_P_MAX_SHIFT;
        }
        if !out_exclusion_threshold_num.is_null() {
            *out_exclusion_threshold_num = protocol::ENCODE_EXCLUSION_THRESHOLD_NUM;
        }
        if !out_rng_seed.is_null() {
            *out_rng_seed = protocol::ENCODE_RNG_SEED;
        }
    }
}

/// Fill the canonical encode area as raw `Fixed` (I4F60 `i64`) bounds — the
/// region the protocol encodes into (`constants.py` `ENCODE_AREA`). Companion
/// to [`bs_encode_params`]; callers pass these straight into the encode/decode
/// area arguments rather than hard-coding the rectangle.
#[no_mangle]
pub extern "C" fn bs_encode_area(
    out_re_min: *mut Fixed,
    out_re_max: *mut Fixed,
    out_im_min: *mut Fixed,
    out_im_max: *mut Fixed,
) {
    let area = crate::protocol::encode_area();
    unsafe {
        if !out_re_min.is_null() {
            *out_re_min = area.re_min;
        }
        if !out_re_max.is_null() {
            *out_re_max = area.re_max;
        }
        if !out_im_min.is_null() {
            *out_im_min = area.im_min;
        }
        if !out_im_max.is_null() {
            *out_im_max = area.im_max;
        }
    }
}

/// Bits encoded per fractal point/stage (`constants.py` `BITS_PER_POINT`).
#[no_mangle]
pub extern "C" fn bs_bits_per_point() -> u32 {
    crate::protocol::BITS_PER_POINT
}

// -----------------------------------------------------------------------
// Viewport leaf-area enumeration
// -----------------------------------------------------------------------
//
// Determines which distinct canonical leaf areas are present in a view, so the
// UX layer can highlight them.  Handle-based because the result is a
// variable-length list: compute once, then read the count and copy each leaf's
// rect + path without re-scanning.

use crate::leaf_enum::{enumerate_leaf_areas, LeafEnumOutcome};

/// Opaque handle holding a leaf-area enumeration result.
pub struct LeafAreasHandle {
    outcome: LeafEnumOutcome,
}

/// Enumerate the distinct leaf areas visible in a view.
///
/// The view is described as in [`bs_render_viewport`]: pixel `(col, row)` maps
/// to `(origin_re + col*step, origin_im + row*step)`.  Sampling steps by
/// `scan_step` pixels on both axes; at most `max_leaves` distinct leaf areas
/// are registered before the result switches to the "too many" status.
///
/// Returns an opaque handle; query it with the `bs_leaf_areas_*` functions and
/// free it with [`bs_leaf_areas_free`].
#[no_mangle]
pub extern "C" fn bs_leaf_areas_compute(
    origin_re: f64,
    origin_im: f64,
    step: f64,
    width_px: u32,
    height_px: u32,
    scan_step: u32,
    max_leaves: u32,
    // Zoom-out guard: max non-excluded decodes before aborting (0 = unbounded).
    max_decodes: u32,
    // Initial (encode) area
    area_re_min: Fixed,
    area_re_max: Fixed,
    area_im_min: Fixed,
    area_im_max: Fixed,
    // Discovery params
    max_iter: u32,
    target_good: u32,
    max_flood_points: u64,
    min_grid_cells: u64,
    p_max_shift: u32,
    exclusion_threshold_num: u32,
    rng_seed: u64,
    num_bits: u32,
    o: u64,
    p: u64,
    q: u64,
    path_prefix: *const u8,
    path_prefix_len: u32,
) -> *mut LeafAreasHandle {
    let prefix = if path_prefix.is_null() || path_prefix_len == 0 {
        "O"
    } else {
        unsafe { std::str::from_utf8_unchecked(slice::from_raw_parts(path_prefix, path_prefix_len as usize)) }
    };
    let area = Rect { re_min: area_re_min, re_max: area_re_max, im_min: area_im_min, im_max: area_im_max };
    let params = DiscoveryParams {
        max_iter,
        min_grid_cells,
        p_max_shift,
        max_flood_points,
        target_good,
        exclusion_threshold_num,
        max_attempts: DEFAULT_MAX_ATTEMPTS,
    };

    let outcome = enumerate_leaf_areas(
        origin_re, origin_im, step, width_px, height_px, scan_step,
        max_leaves as usize, max_decodes as usize, area, &params, rng_seed,
        num_bits as usize, o, p, q, prefix,
    );
    Box::into_raw(Box::new(LeafAreasHandle { outcome }))
}

/// Result status: `0` = a list of leaf areas is available (query with
/// [`bs_leaf_areas_count`] / [`bs_leaf_areas_rect`] / [`bs_leaf_areas_path`]);
/// `1` = more than `max_leaves` distinct leaf areas are present; `2` = the
/// decode budget was hit before finishing (zoom-out).  For both `1` and `2`
/// the UX should prompt the user to zoom in.
#[no_mangle]
pub extern "C" fn bs_leaf_areas_status(handle: *const LeafAreasHandle) -> i32 {
    let h = unsafe { &*handle };
    match h.outcome {
        LeafEnumOutcome::Leaves(_) => 0,
        LeafEnumOutcome::TooMany { .. } => 1,
        LeafEnumOutcome::BudgetExhausted { .. } => 2,
    }
}

/// Number of registered leaf areas (0 unless the status is `0`).
#[no_mangle]
pub extern "C" fn bs_leaf_areas_count(handle: *const LeafAreasHandle) -> u32 {
    let h = unsafe { &*handle };
    match &h.outcome {
        LeafEnumOutcome::Leaves(l) => l.len() as u32,
        _ => 0,
    }
}

/// Write leaf `idx`'s rectangle as Fixed `[re_min, re_max, im_min, im_max]`
/// into `out_rect` (4 elements).  No-op if `idx` is out of range.
#[no_mangle]
pub extern "C" fn bs_leaf_areas_rect(
    handle: *const LeafAreasHandle,
    idx: u32,
    out_rect: *mut Fixed,
) {
    let h = unsafe { &*handle };
    if let LeafEnumOutcome::Leaves(leaves) = &h.outcome {
        if let Some(leaf) = leaves.get(idx as usize) {
            unsafe {
                let rect = slice::from_raw_parts_mut(out_rect, 4);
                rect[0] = leaf.rect.re_min;
                rect[1] = leaf.rect.re_max;
                rect[2] = leaf.rect.im_min;
                rect[3] = leaf.rect.im_max;
            }
        }
    }
}

/// Copy leaf `idx`'s bisection path (its canonical identity) into `out_buf` as
/// a NUL-terminated UTF-8 string, writing up to `buf_len` bytes including the
/// terminator.  Returns the full path length in bytes (excluding NUL), even if
/// truncated; returns 0 for an out-of-range index.
#[no_mangle]
pub extern "C" fn bs_leaf_areas_path(
    handle: *const LeafAreasHandle,
    idx: u32,
    out_buf: *mut u8,
    buf_len: u32,
) -> u32 {
    let h = unsafe { &*handle };
    let path_bytes = match &h.outcome {
        LeafEnumOutcome::Leaves(leaves) => match leaves.get(idx as usize) {
            Some(leaf) => leaf.path.as_bytes(),
            None => return 0,
        },
        _ => return 0,
    };
    if !out_buf.is_null() && buf_len > 0 {
        let copy_len = std::cmp::min(path_bytes.len(), (buf_len - 1) as usize);
        let out = unsafe { slice::from_raw_parts_mut(out_buf, buf_len as usize) };
        out[..copy_len].copy_from_slice(&path_bytes[..copy_len]);
        out[copy_len] = 0; // NUL terminator
    }
    path_bytes.len() as u32
}

/// Free a leaf-areas handle.
#[no_mangle]
pub extern "C" fn bs_leaf_areas_free(handle: *mut LeafAreasHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

// -----------------------------------------------------------------------
// Canonical island (per-leaf visualisation shape)
// -----------------------------------------------------------------------
//
// For a leaf area, selects its canonical island and exposes the island's
// flood points so the UX can draw the shape (the union of pixel_delta cells).
// Handle-based because the point set is variable-length: compute once, read the
// metadata, then copy the points.

use crate::discovery::{canonical_island, CanonicalIsland};

/// Opaque handle holding a canonical-island result (`None` if no island found).
pub struct CanonicalIslandHandle {
    island: Option<CanonicalIsland>,
}

/// Select the canonical island of a leaf rectangle (`bs_canonical_island_*`).
///
/// `path` should be the leaf's bisection path (from `bs_decode_full` /
/// `bs_leaf_areas_path`), and `(o, p, q)` the same fractal the leaf was decoded
/// on.  The discovery params are a visualisation choice (resolution / flood
/// cap), independent of the encoder's.  Returns an opaque handle; query it with
/// the other `bs_canonical_island_*` functions and free with
/// [`bs_canonical_island_free`].
#[no_mangle]
pub extern "C" fn bs_canonical_island_compute(
    area_re_min: Fixed,
    area_re_max: Fixed,
    area_im_min: Fixed,
    area_im_max: Fixed,
    max_iter: u32,
    target_good: u32,
    max_flood_points: u64,
    min_grid_cells: u64,
    p_max_shift: u32,
    exclusion_threshold_num: u32,
    rng_seed: u64,
    o: u64,
    p: u64,
    q: u64,
    path: *const u8,
    path_len: u32,
) -> *mut CanonicalIslandHandle {
    let path_str = if path.is_null() || path_len == 0 {
        "O"
    } else {
        unsafe { std::str::from_utf8_unchecked(slice::from_raw_parts(path, path_len as usize)) }
    };
    let area = Rect { re_min: area_re_min, re_max: area_re_max, im_min: area_im_min, im_max: area_im_max };
    let params = DiscoveryParams {
        max_iter,
        min_grid_cells,
        p_max_shift,
        max_flood_points,
        target_good,
        exclusion_threshold_num,
        max_attempts: DEFAULT_MAX_ATTEMPTS,
    };
    let island = canonical_island(&area, &params, rng_seed, o, p, q, path_str);
    Box::into_raw(Box::new(CanonicalIslandHandle { island }))
}

/// `1` if a canonical island was found, `0` otherwise.
#[no_mangle]
pub extern "C" fn bs_canonical_island_found(handle: *const CanonicalIslandHandle) -> i32 {
    let h = unsafe { &*handle };
    if h.island.is_some() { 1 } else { 0 }
}

/// Write the island metadata: escape count, pixel (cell) count, `pixel_delta`
/// grid spacing (Fixed), and bounding box `[re_min, re_max, im_min, im_max]`
/// (Fixed).  Any out pointer may be null to skip it; all are left untouched if
/// no island was found.
#[no_mangle]
pub extern "C" fn bs_canonical_island_meta(
    handle: *const CanonicalIslandHandle,
    out_escape: *mut u32,
    out_pixel_count: *mut u64,
    out_pixel_delta: *mut Fixed,
    out_bbox: *mut Fixed,
) {
    let h = unsafe { &*handle };
    if let Some(island) = &h.island {
        unsafe {
            if !out_escape.is_null() { *out_escape = island.escape_count; }
            if !out_pixel_count.is_null() { *out_pixel_count = island.pixel_count; }
            if !out_pixel_delta.is_null() { *out_pixel_delta = island.pixel_delta; }
            if !out_bbox.is_null() {
                let bbox = slice::from_raw_parts_mut(out_bbox, 4);
                bbox[0] = island.bbox.re_min;
                bbox[1] = island.bbox.re_max;
                bbox[2] = island.bbox.im_min;
                bbox[3] = island.bbox.im_max;
            }
        }
    }
}

/// Number of flood points in the canonical island (0 if none found).
#[no_mangle]
pub extern "C" fn bs_canonical_island_num_points(handle: *const CanonicalIslandHandle) -> u32 {
    let h = unsafe { &*handle };
    match &h.island {
        Some(island) => island.points.len() as u32,
        None => 0,
    }
}

/// Copy the island's flood points into `out_points` as interleaved Fixed pairs
/// `[re0, im0, re1, im1, ...]`.  `max_points` is the capacity in *points*
/// (`out_points` must hold `2 * max_points` Fixed).  Returns the number of
/// points actually written (`min(num_points, max_points)`).
#[no_mangle]
pub extern "C" fn bs_canonical_island_points(
    handle: *const CanonicalIslandHandle,
    out_points: *mut Fixed,
    max_points: u32,
) -> u32 {
    let h = unsafe { &*handle };
    let island = match &h.island {
        Some(island) => island,
        None => return 0,
    };
    if out_points.is_null() || max_points == 0 {
        return 0;
    }
    let n = island.points.len().min(max_points as usize);
    let out = unsafe { slice::from_raw_parts_mut(out_points, n * 2) };
    for (i, &(re, im)) in island.points.iter().take(n).enumerate() {
        out[i * 2] = re;
        out[i * 2 + 1] = im;
    }
    n as u32
}

/// Free a canonical-island handle.
#[no_mangle]
pub extern "C" fn bs_canonical_island_free(handle: *mut CanonicalIslandHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

// ---------------------------------------------------------------------------
// Orbit protocol (0.4.0) — engine-authoritative primitives.
//
// These expose the orbit derivation (crate::orbit) and the GF(2^32) share
// layer (crate::shamir) over the C ABI so callers read the protocol from the
// engine rather than re-implementing it.  They do not touch the 0.3.0
// encode/decode pipeline; protocol.py is flipped to drive through them in a
// later step (see great-wall-docs/next-steps/core-orbit-redesign-plan.md §5).
// ---------------------------------------------------------------------------

/// `o_0 = H(σ)` — orbit root from the public Namtso salt. Writes 32 bytes.
#[no_mangle]
pub extern "C" fn bs_orbit_root(sigma_ptr: *const u8, sigma_len: u32, out_digest: *mut u8) {
    let sigma = unsafe { slice::from_raw_parts(sigma_ptr, sigma_len as usize) };
    let digest = crate::orbit::orbit_root(sigma);
    unsafe { slice::from_raw_parts_mut(out_digest, 32).copy_from_slice(&digest) };
}

/// `theta_i_j = H(o_i ‖ j)` — parameter digest of board `j` at orbit point `o_i`
/// (`o_ptr` = 32 bytes). Writes the 32-byte digest; `(o,p,q)` attribution is
/// applied downstream, unchanged.
#[no_mangle]
pub extern "C" fn bs_theta(o_ptr: *const u8, j: u32, out_digest: *mut u8) {
    let mut o = [0u8; 32];
    o.copy_from_slice(unsafe { slice::from_raw_parts(o_ptr, 32) });
    let digest = crate::orbit::theta_digest(&o, j);
    unsafe { slice::from_raw_parts_mut(out_digest, 32).copy_from_slice(&digest) };
}

/// `K_i = H(o_i ‖ Sh_i)` — per-stage master secret (`i > 0`), the cheap-`H`
/// commitment. `o_ptr` = 32 bytes; `sh_ptr`/`sh_len` = serialized Shamir
/// polynomial. Writes 32 bytes.
#[no_mangle]
pub extern "C" fn bs_master_secret(
    o_ptr: *const u8,
    sh_ptr: *const u8,
    sh_len: u32,
    out_digest: *mut u8,
) {
    let mut o = [0u8; 32];
    o.copy_from_slice(unsafe { slice::from_raw_parts(o_ptr, 32) });
    let sh = unsafe { slice::from_raw_parts(sh_ptr, sh_len as usize) };
    let digest = crate::orbit::master_secret(&o, sh);
    unsafe { slice::from_raw_parts_mut(out_digest, 32).copy_from_slice(&digest) };
}

/// One orbit step: writes `K_i = H(o_i ‖ Sh_i)` to `out_k` (32 bytes) and
/// `o_{i+1} = H*(K_i)` (Argon2d `profile`, `steps` passes) to `out_next` (32
/// bytes). The raw `{o_i, Sh_i}` copies are zeroized before the memory-hard
/// step (crate::orbit::orbit_step). `profile`: 0=Basic, 1=Advanced, 2=Great Wall.
#[no_mangle]
pub extern "C" fn bs_orbit_advance(
    o_ptr: *const u8,
    sh_ptr: *const u8,
    sh_len: u32,
    steps: u32,
    profile: u8,
    out_k: *mut u8,
    out_next: *mut u8,
) {
    use zeroize::Zeroizing;
    let mut o = Zeroizing::new([0u8; 32]);
    o.copy_from_slice(unsafe { slice::from_raw_parts(o_ptr, 32) });
    let sh = Zeroizing::new(unsafe { slice::from_raw_parts(sh_ptr, sh_len as usize) }.to_vec());
    let (k_i, o_next) = crate::orbit::orbit_step(o, sh, steps, profile);
    unsafe {
        slice::from_raw_parts_mut(out_k, 32).copy_from_slice(&k_i);
        slice::from_raw_parts_mut(out_next, 32).copy_from_slice(&o_next);
    }
}

/// Interpolate the full Shamir polynomial `Sh` from `t` points over GF(2^32):
/// `xs`/`ys` are `t` `u32` abscissae/points; writes `t` `u32` coefficients
/// (ascending powers) to `out_sh`. `Sh` is `t·32` bits — NOT the constant term.
#[no_mangle]
pub extern "C" fn bs_shamir_interp(
    xs_ptr: *const u32,
    ys_ptr: *const u32,
    t: u32,
    out_sh: *mut u32,
) {
    let n = t as usize;
    let xs = unsafe { slice::from_raw_parts(xs_ptr, n) };
    let ys = unsafe { slice::from_raw_parts(ys_ptr, n) };
    let sh = crate::shamir::interpolate(xs, ys);
    unsafe { slice::from_raw_parts_mut(out_sh, n).copy_from_slice(&sh) };
}

/// Canonical per-stage Shamir thresholds `t_i` for a setup `level` (1-based):
/// index 0 is stage 0, `1..=N` the deep stages (`t_i == r_i`, `t_i·32` bits).
///
/// Query-then-fill (like `bs_salt_pepper_canonicalize`): pass `out = null` /
/// `cap = 0` to learn the count `N+1`, then call again with a `u32` buffer of
/// that size. Always returns the full count. Returns 0 for an invalid `level`
/// (`0` or `> MAX_SETUP_LEVEL`) so the ABI never panics.
#[no_mangle]
pub extern "C" fn bs_setup_tier_thresholds(level: u32, out: *mut u32, cap: u32) -> u32 {
    if level == 0 || level > crate::setup_tiers::MAX_SETUP_LEVEL {
        return 0;
    }
    let t = crate::setup_tiers::thresholds(level);
    if !out.is_null() && cap > 0 {
        let n = t.len().min(cap as usize);
        let o = unsafe { slice::from_raw_parts_mut(out, cap as usize) };
        o[..n].copy_from_slice(&t[..n]);
    }
    t.len() as u32
}

/// Whether a setup `level` is **substandard** (the entry tier, Setup 1, with a
/// 64-bit deep stage). Returns 1 if substandard, else 0 (0 also for an invalid
/// `level`). Interfaces MUST surface this loudly wherever Setup 1 is offered.
#[no_mangle]
pub extern "C" fn bs_setup_tier_substandard(level: u32) -> u8 {
    if level == 0 || level > crate::setup_tiers::MAX_SETUP_LEVEL {
        return 0;
    }
    crate::setup_tiers::is_substandard(level) as u8
}

