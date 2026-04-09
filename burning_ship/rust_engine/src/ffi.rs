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
const DEFAULT_MAX_ATTEMPTS: u64 = 500_000;

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

use crate::argon2_hash::{iterative_argon2, argon2_single};

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

