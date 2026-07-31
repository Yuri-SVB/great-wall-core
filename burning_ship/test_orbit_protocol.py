"""Round-trip test for the orbit-protocol (0.4.0) additive path in protocol.py.

Validates encode_orbit <-> decode_orbit (entropy -> points -> entropy) and K_i
consistency across setup tiers, using a CHEAP deterministic advance so the
orchestration is exercised without the memory-hard Argon2 advance. Encode and
decode use the SAME advance, so the bijection holds regardless of which advance
is used; the real memory-hard advance is validated at the Rust level
(orbit_step_with) and via the FFI. Requires `cargo build --release` in
rust_engine/ first.

Run:  python3 test_orbit_protocol.py   (from burning_ship/)
"""
import hashlib
import random

import protocol
import burning_ship_engine as eng
from burning_ship_engine import DiscoveryParams, Rect
from constants import BITS_PER_POINT

# Light discovery params + smaller area so the round-trip runs quickly (the same
# config test_bijection.py uses). Encode and decode use the SAME params, so the
# bijection is unaffected; production uses GUI_PARAMS / ENCODE_AREA by default.
FAST_PARAMS = DiscoveryParams(
    max_iter=200, target_good=30, max_flood_points=10000,
    min_grid_cells=1024, p_max_shift=1, exclusion_threshold_num=204, rng_seed=0x42,
)
TEST_AREA = Rect.from_f64(-2.0, 1.0, -1.5, 1.0)


def cheap_advance(o_bytes, sh_bytes):
    """Deterministic, fast stand-in for o_{i+1} = H*(K_i)."""
    return hashlib.sha256(o_bytes + sh_bytes + b"ORBIT-ADVANCE").digest()


def rand_chunk(rng):
    return [rng.randrange(2) for _ in range(BITS_PER_POINT)]


def check(cond, msg):
    if not cond:
        raise AssertionError("FAILED: " + msg)
    print(f"  ok: {msg}")


def round_trip(setup_level, rng):
    thresholds = eng.setup_tier_thresholds(setup_level)
    stage_chunks = [[rand_chunk(rng) for _ in range(t)] for t in thresholds]
    sigma = bytes(rng.randrange(256) for _ in range(128))

    stages, k_enc = protocol.encode_orbit(
        sigma, setup_level, stage_chunks, advance_fn=cheap_advance,
        params=FAST_PARAMS, area=TEST_AREA)
    stage_points = [list(st.points) for st in stages]

    recovered, k_dec = protocol.decode_orbit(
        sigma, stage_points, setup_level, advance_fn=cheap_advance,
        params=FAST_PARAMS, area=TEST_AREA)

    sub = " (SUBSTANDARD)" if eng.setup_tier_substandard(setup_level) else ""
    check(recovered == stage_chunks,
          f"level {setup_level} {thresholds}{sub}: entropy round-trips")
    check(k_enc == k_dec, f"level {setup_level}: K matches on encode & decode")
    check(len(stages[-1].sh) == thresholds[-1],
          f"level {setup_level}: terminal Sh has t_N={thresholds[-1]} coeffs "
          f"({thresholds[-1] * 32} bits)")
    return k_enc


def main():
    rng = random.Random(0x0B17)
    print("orbit encode/decode round-trip across setup tiers:")
    for level in (1, 2, 3):
        round_trip(level, rng)

    print("K_i determinism & sensitivity:")
    rng2 = random.Random(777)
    thr = eng.setup_tier_thresholds(2)
    sc = [[rand_chunk(rng2) for _ in range(t)] for t in thr]
    sig = bytes(rng2.randrange(256) for _ in range(128))
    _, k1 = protocol.encode_orbit(sig, 2, sc, advance_fn=cheap_advance,
                                  params=FAST_PARAMS, area=TEST_AREA)
    _, k2 = protocol.encode_orbit(sig, 2, sc, advance_fn=cheap_advance,
                                  params=FAST_PARAMS, area=TEST_AREA)
    check(k1 == k2, "K is deterministic for identical (sigma, points)")
    sig2 = bytes([sig[0] ^ 0x01]) + sig[1:]
    _, k3 = protocol.encode_orbit(sig2, 2, sc, advance_fn=cheap_advance,
                                  params=FAST_PARAMS, area=TEST_AREA)
    check(k3 != k1, "distinct sigma -> distinct K (orbit re-rooted)")

    print("\nALL ORBIT PROTOCOL ROUND-TRIP TESTS PASSED")


if __name__ == "__main__":
    main()
