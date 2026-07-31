"""Generate interim smoke vectors for the 0.4.0 orbit protocol.

core-orbit-redesign-plan.md §5 step 7 / §4: at least one interim smoke vector per
setup tier — (sigma, setup-tier, D) -> {theta_i_j, p_i_j, Sh_i, o_i, K}. These
pin reproducibility of the orbit derivation. Generation runs the REAL
memory-hard advance (slow, Argon2d) so the o_i chain is genuine; the committed
verifier (test_orbit_vectors.py) checks the cheap per-stage derivations against
the frozen o_i without re-running the advance.

Run:  python3 generate_orbit_vectors.py   (from burning_ship/; slow — real Argon2)
"""
import json
import os
import random

import protocol
import burning_ship_engine as eng
from burning_ship_engine import DiscoveryParams, Rect, get_engine_version
from constants import BITS_PER_POINT

# Frozen discovery config for the vectors (stored so the verifier reproduces
# exactly; a light config keeps point encode/decode fast — the advance dominates).
VEC_PARAMS = dict(max_iter=200, target_good=30, max_flood_points=10000,
                  min_grid_cells=1024, p_max_shift=1, exclusion_threshold_num=204,
                  rng_seed=0x42)
VEC_AREA = (-2.0, 1.0, -1.5, 1.0)  # re_min, re_max, im_min, im_max
PROFILE = 0        # Basic
ITERATIONS = 1     # D = 1 advance pass per stage transition
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "test_vectors", "orbit-v0.4.0")


def _hex_i64(v):
    return format(v & 0xFFFFFFFFFFFFFFFF, "016x")


def build(level, seed):
    rng = random.Random(seed)
    thresholds = eng.setup_tier_thresholds(level)
    stage_chunks = [[[rng.randrange(2) for _ in range(BITS_PER_POINT)]
                     for _ in range(t)] for t in thresholds]
    sigma = bytes(rng.randrange(256) for _ in range(128))

    params = DiscoveryParams(**VEC_PARAMS)
    area = Rect.from_f64(*VEC_AREA)
    stages, k = protocol.encode_orbit(
        sigma, level, stage_chunks, iterations=ITERATIONS, profile=PROFILE,
        params=params, area=area)

    stage_docs = []
    for st, chunks in zip(stages, stage_chunks):
        stage_docs.append({
            "index": st.index,
            "threshold": st.threshold,
            "o_hex": st.o_bytes.hex(),
            "chunks_u32": [format(protocol._bits_to_u32(c), "08x") for c in chunks],
            "fractals": [{"o": _hex_i64(o), "p": _hex_i64(p), "q": _hex_i64(q)}
                         for (o, p, q) in st.fractals],
            "points": [{"re_raw": _hex_i64(re), "im_raw": _hex_i64(im)}
                       for (re, im) in st.points],
            "sh": [format(c, "08x") for c in st.sh],
            "k_hex": st.k.hex(),
        })

    return {
        "engine_version": get_engine_version(),
        "protocol_version": protocol.PROTOCOL_VERSION,
        "sigma_hex": sigma.hex(),
        "setup_level": level,
        "substandard": bool(eng.setup_tier_substandard(level)),
        "thresholds": thresholds,
        "argon2_profile": PROFILE,
        "iterations": ITERATIONS,
        "vec_params": VEC_PARAMS,
        "vec_area": VEC_AREA,
        "terminal_k_hex": k.hex(),
        "stages": stage_docs,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for level in (1, 2, 3):
        print(f"generating orbit smoke vector: setup {level} "
              f"({eng.setup_tier_thresholds(level)}) — real advance, please wait...")
        vec = build(level, seed=0x0B17 + level)
        path = os.path.join(OUT_DIR, f"orbit_setup{level}.json")
        with open(path, "w") as f:
            json.dump(vec, f, indent=2)
        print(f"  wrote {path}  (terminal K={vec['terminal_k_hex'][:16]}...)")


if __name__ == "__main__":
    main()
