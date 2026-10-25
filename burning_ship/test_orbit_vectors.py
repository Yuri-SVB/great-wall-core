"""Verify the frozen 0.5.0 orbit smoke vectors (test_vectors/orbit-v0.5.0/).

FAST: reproduces the orbit's cheap per-stage derivations — theta_i_j -> (o,p,q),
point decode, Sh_i, u_i, K_i, K_i^L — from the FROZEN o_i in each vector, so this
does NOT re-run the memory-hard advance. The advance chain (o_{i+1} = H*(u_i))
was exercised when the vectors were generated (generate_orbit_vectors.py);
`--full` re-runs it, one H* pass at a time from the frozen u_i (slow: Argon2d at
the vector's profile, >= 1 GiB).

STALE guard mirrors test_vectors.py: a vector whose engine/protocol version does
not match the current build is reported STALE and skipped (never a false pass).

Run:  python3 test_orbit_vectors.py [--full]   (from burning_ship/)
"""
import glob
import json
import os
import sys

import protocol
import burning_ship_engine as eng
from burning_ship_engine import decode_full, DiscoveryParams, Rect, get_engine_version
from constants import BITS_PER_POINT

VEC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "test_vectors", "orbit-v0.5.0")


def _from_hex_i64(h):
    """Parse 16-hex unsigned back to a signed i64."""
    v = int(h, 16)
    return v - (1 << 64) if v >= (1 << 63) else v


def verify_vector(path, full=False):
    name = os.path.basename(path)
    with open(path) as f:
        vec = json.load(f)

    # STALE guard: engine + protocol version must match the current build.
    cur_engine, cur_proto = get_engine_version(), protocol.PROTOCOL_VERSION
    if vec["engine_version"] != cur_engine or vec["protocol_version"] != cur_proto:
        print(f"  STALE {name} (engine {vec['engine_version']}/protocol "
              f"{vec['protocol_version']} != current {cur_engine}/{cur_proto})")
        return None  # skip

    params = DiscoveryParams(**vec["vec_params"])
    area = Rect.from_f64(*vec["vec_area"])

    failures = []
    last_k = None
    advances = []  # (stage index, u_i) for --full
    for st in sorted(vec["stages"], key=lambda s: s["index"]):
        o_i = bytes.fromhex(st["o_hex"])
        t = st["threshold"]
        xs, ys = [], []
        for j in range(t):
            # theta_i_j -> (o,p,q) reproduces the stored fractal params
            op, pp, qp = protocol._orbit_params(o_i, j)
            got = {"o": format(op, "016x"), "p": format(pp, "016x"),
                   "q": format(qp, "016x")}
            if got != st["fractals"][j]:
                failures.append(f"stage {st['index']} fractal {j}: theta params differ")
            # stored point decodes to the stored 32-bit value
            re_raw = _from_hex_i64(st["points"][j]["re_raw"])
            im_raw = _from_hex_i64(st["points"][j]["im_raw"])
            bits, _leaf, _valid, _path = decode_full(
                re_raw, im_raw, BITS_PER_POINT, area=area, params=params,
                o=op, p=pp, q=qp, path_prefix="O")
            y = protocol._bits_to_u32(list(bits))
            if format(y, "08x") != st["chunks_u32"][j]:
                failures.append(f"stage {st['index']} fractal {j}: decoded value differs")
            xs.append(j + 1)
            ys.append(y)
        # Sh_i and K_i reproduce
        sh = eng.shamir_interp(xs, ys)
        if [format(c, "08x") for c in sh] != st["sh"]:
            failures.append(f"stage {st['index']}: Sh differs")
        sh_bytes = eng.sh_to_bytes(sh)
        u = eng.orbit_commitment(o_i, sh_bytes)
        if u.hex() != st["u_hex"]:
            failures.append(f"stage {st['index']}: u_i differs")
        k = eng.master_secret(o_i, sh_bytes)
        if k.hex() != st["k_hex"]:
            failures.append(f"stage {st['index']}: K_i differs")
        for ex in st["exports"]:
            key = eng.export_key(k, ex["label"].encode("ascii"))
            if key.hex() != ex["key_hex"]:
                failures.append(f"stage {st['index']}: K_i^L differs "
                                f"for label {ex['label']!r}")
        last_k = st["k_hex"]
        advances.append((st["index"], u))

    if full:
        # o_{i+1} = H*^D(u_i): re-run the memory-hard advance from each frozen
        # u_i and compare with the next stage's frozen o.
        o_by_index = {st["index"]: st["o_hex"] for st in vec["stages"]}
        for index, u in advances:
            if index + 1 not in o_by_index:
                continue
            digest = u
            for _ in range(vec["iterations"]):
                digest = eng.orbit_argon2_single(digest, vec["argon2_profile"])
            if digest.hex() != o_by_index[index + 1]:
                failures.append(f"stage {index}: H*(u_i) != frozen o_{index + 1}")

    if last_k != vec["terminal_k_hex"]:
        failures.append("terminal K != last stage K")

    if failures:
        print(f"  FAIL  {name}: {len(failures)} difference(s)")
        for msg in failures[:8]:
            print(f"        {msg}")
        return False
    sub = " (substandard)" if vec["substandard"] else ""
    print(f"  OK    {name}  setup {vec['setup_level']} {vec['thresholds']}{sub}")
    return True


def main():
    full = "--full" in sys.argv[1:]
    paths = sorted(glob.glob(os.path.join(VEC_DIR, "orbit_setup*.json")))
    if not paths:
        print(f"no orbit vectors found in {VEC_DIR}")
        raise SystemExit(1)
    print(f"=== Orbit smoke vectors (engine {get_engine_version()} / protocol "
          f"{protocol.PROTOCOL_VERSION}) ===")
    passed = failed = stale = 0
    for p in paths:
        r = verify_vector(p, full=full)
        if r is None:
            stale += 1
        elif r:
            passed += 1
        else:
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed, {stale} STALE-skipped")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
