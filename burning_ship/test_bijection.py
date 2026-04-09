#!/usr/bin/env python3
"""
End-to-end test: encode a small bit array, decode it back, verify bijection.

Usage: python3 test_bijection.py
"""

import sys
import time

# Add parent dir to path
sys.path.insert(0, '.')

from burning_ship_engine import encode, decode, get_precision, DiscoveryParams, Rect, DEFAULT_AREA, fixed_to_f64


def test_bijection(bits, label="", params=None, area=None):
    """Encode bits, decode the result, verify round-trip."""
    print(f"\n{'='*60}")
    print(f"Test: {label}")
    print(f"Input bits:  {bits}")
    print(f"Num bits:    {len(bits)}")

    t0 = time.time()
    result = encode(bits, area=area, params=params)
    t_encode = time.time() - t0

    print(f"Encoded to:  ({result.point_re:.15f}, {result.point_im:.15f})")
    print(f"Raw i64:     ({result.point_re_raw}, {result.point_im_raw})")
    print(f"Bits consumed: {result.bits_consumed}")
    print(f"Final rect:  {result.final_rect}")
    print(f"Encode time: {t_encode:.3f}s")

    # Print step details
    for i in range(min(result.num_steps, 20)):
        step = result.get_step(i)
        axis = "V" if step['split_vertical'] else "H"
        side = "L" if step['chose_larger'] else "S"
        print(f"  step {i:2d}: bit={step['bit']} axis={axis} "
              f"side={side} f={step['contraction']:.3f} "
              f"islands={step['islands_found']} "
              f"split={step['split_coord_f64']:.6f}")
    if result.num_steps > 20:
        print(f"  ... ({result.num_steps - 20} more steps)")

    t0 = time.time()
    decoded = decode(result.point_re_raw, result.point_im_raw, len(bits),
                     area=area, params=params)
    t_decode = time.time() - t0

    print(f"Decoded bits: {decoded}")
    print(f"Decode time:  {t_decode:.3f}s")

    match = bits == decoded
    print(f"MATCH: {'YES' if match else 'NO <<<< FAILURE'}")
    return match


def main():
    frac_bits, int_bits = get_precision()
    print(f"Fixed-point precision: I{int_bits+1}F{frac_bits}")
    print(f"  (1 sign + {int_bits} integer + {frac_bits} fractional = 64 bits)")

    # Use faster params for testing
    fast_params = DiscoveryParams(
        max_iter=200,
        target_good=30,
        max_flood_points=10000,
        min_grid_cells=1024, # p0 ≈ sqrt(area / 1024)
        p_max_shift=1,      # ~1/2 of area width
        exclusion_threshold_num=204,  # 204/256 = 0.797
        rng_seed=0x42,
    )

    # Smaller area for faster discovery
    test_area = Rect.from_f64(-2.0, 1.0, -1.5, 1.0)

    all_pass = True

    # Test 1: Single bit
    all_pass &= test_bijection([0], "single bit 0", fast_params, test_area)
    all_pass &= test_bijection([1], "single bit 1", fast_params, test_area)

    # Test 2: Two bits
    for bits in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        all_pass &= test_bijection(bits, f"two bits {bits}", fast_params, test_area)

    # Test 3: Four bits
    all_pass &= test_bijection([1, 0, 1, 1], "four bits [1,0,1,1]", fast_params, test_area)

    # Test 4: Eight bits
    all_pass &= test_bijection([1, 0, 0, 1, 1, 0, 1, 0], "eight bits", fast_params, test_area)

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
