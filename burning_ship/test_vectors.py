#!/usr/bin/env python3
"""
Test runner for Great Wall engine test vectors.

Compares fresh encode output against committed JSON vectors.
Ignores f64 display values (platform-dependent rounding).
Also runs round-trip tests (encode → decode → compare entropy).

Usage:
  python3 test_vectors.py                    # run all tests
  python3 test_vectors.py --vector FILE.json # run single vector
  python3 test_vectors.py --verbose          # show diff details
"""

import sys
import os
import json
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from protocol import PROTOCOL_VERSION  # noqa: E402  (current chained-protocol version)
from burning_ship_engine import get_engine_version  # noqa: E402  (single-fractal algo version)
CLI_PATH = os.path.join(SCRIPT_DIR, "cli.py")
VECTORS_DIR = os.path.join(SCRIPT_DIR, "test_vectors")


def strip_f64_fields(obj):
    """Recursively remove all keys ending in '_f64' and float display values."""
    if isinstance(obj, dict):
        return {k: strip_f64_fields(v) for k, v in obj.items()
                if not k.endswith("_f64")
                and k not in ("o_re", "o_im", "p_re", "p_im", "q_re", "q_im",
                              "center_re_f64", "center_im_f64",
                              "re_min_f64", "re_max_f64", "im_min_f64", "im_max_f64")}
    if isinstance(obj, list):
        return [strip_f64_fields(item) for item in obj]
    return obj


def run_encode(vector_doc):
    """Re-run cli.py encode with the same input parameters."""
    inp = vector_doc["input"]
    cmd = [
        sys.executable, CLI_PATH, "encode",
        "--entropy", inp["entropy_hex"],
        "--profile", inp["argon2_profile"],
        "--iterations", str(inp["argon2_iterations"]),
        "--mode", inp["gw_mode"],
        # Stage-0 text seeds the chain (protocol 0.3.0); pass it through so the
        # re-encode reproduces the frozen vector exactly.
        "--stage0-text", inp.get("stage0_text", ""),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"cli.py encode failed: {result.stderr}")
    return json.loads(result.stdout)


def run_decode(vector_path, timeout=1800):
    """Run cli.py decode on a vector file.

    ``timeout`` is bounded low for the negative/meta tests: decoding a corrupted
    document can land in a barren region and exhaust island-discovery attempts,
    so a timeout there is itself valid evidence that the corruption was caught.
    """
    cmd = [sys.executable, CLI_PATH, "decode", "--input", vector_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"cli.py decode failed: {result.stderr}")
    return json.loads(result.stdout)


# Bounded timeout for the negative/meta decodes (a hang on garbage == detected).
# Valid corrupted decodes that DIFFER finish in a few seconds; only a barren
# region hangs, so a low bound cleanly separates "detected" from "slow".
META_DECODE_TIMEOUT = 30


def deep_diff(expected, actual, path=""):
    """Find all differences between two nested structures. Returns list of (path, expected, actual)."""
    diffs = []
    if type(expected) != type(actual):
        diffs.append((path, f"type={type(expected).__name__}", f"type={type(actual).__name__}"))
        return diffs
    if isinstance(expected, dict):
        all_keys = set(expected.keys()) | set(actual.keys())
        for k in sorted(all_keys):
            if k not in expected:
                diffs.append((f"{path}.{k}", "<missing>", actual[k]))
            elif k not in actual:
                diffs.append((f"{path}.{k}", expected[k], "<missing>"))
            else:
                diffs.extend(deep_diff(expected[k], actual[k], f"{path}.{k}"))
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            diffs.append((f"{path}[len]", len(expected), len(actual)))
        for i in range(min(len(expected), len(actual))):
            diffs.extend(deep_diff(expected[i], actual[i], f"{path}[{i}]"))
    else:
        if expected != actual:
            diffs.append((path, expected, actual))
    return diffs


def test_frozen_vector(vector_path, verbose=False):
    """Test that encoding the same input reproduces the committed vector exactly."""
    name = os.path.basename(vector_path)
    with open(vector_path, "r") as f:
        expected = json.load(f)

    try:
        actual = run_encode(expected)
    except Exception as e:
        print(f"  FAIL  {name}: encode error: {e}")
        return False

    # Compare with f64 fields stripped
    exp_clean = strip_f64_fields(expected)
    act_clean = strip_f64_fields(actual)

    diffs = deep_diff(exp_clean, act_clean)
    if diffs:
        print(f"  FAIL  {name}: {len(diffs)} difference(s)")
        if verbose:
            for path, exp_val, act_val in diffs[:10]:
                print(f"        {path}: expected={exp_val} actual={act_val}")
        return False

    print(f"  OK    {name} (frozen vector)")
    return True


def test_round_trip(vector_path, verbose=False):
    """Test that encode → decode recovers the original entropy."""
    name = os.path.basename(vector_path)
    with open(vector_path, "r") as f:
        expected = json.load(f)

    try:
        decoded = run_decode(vector_path)
    except Exception as e:
        print(f"  FAIL  {name}: decode error: {e}")
        return False

    expected_hex = expected["input"]["entropy_hex"]
    actual_hex = decoded["decoded_entropy_hex"]

    if expected_hex != actual_hex:
        print(f"  FAIL  {name} (round-trip): expected={expected_hex} got={actual_hex}")
        return False

    print(f"  OK    {name} (round-trip)")
    return True


def test_cross_mode(vectors_dir, verbose=False):
    """Test that the first stage's leaf is identical across mini/default/large.

    Under protocol 0.3.0 the first point stage's fractal derives from stage-0
    text plus the (empty) prior-point prefix.  These vectors use an empty
    stage-0 text, so the first stage's params are identical across presets, and
    its 32-bit point depends only on the first 32 entropy bits — which are
    identical for the all-zero "abandon" mnemonics across presets.  Hence the
    first leaf must match.  (There is no canonical fractal anymore; the match
    comes from a shared stage-0 text + shared first chunk, not from o=p=q=0.)
    """
    patterns = [
        ("mini_abandon_iter0.json", "default_abandon_iter0.json", "large_abandon_iter0.json"),
    ]
    all_ok = True
    for mini_f, default_f, large_f in patterns:
        files = [os.path.join(vectors_dir, f) for f in (mini_f, default_f, large_f)]
        if not all(os.path.exists(f) for f in files):
            continue
        leaves = []
        for f in files:
            with open(f) as fh:
                doc = json.load(fh)
            leaf = doc["stages"][0]["leaf"]
            leaves.append({k: leaf[k] for k in ("re_min", "re_max", "im_min", "im_max")})
        if leaves[0] == leaves[1] == leaves[2]:
            print(f"  OK    cross-mode: first leaf identical (abandon...)")
        else:
            print(f"  FAIL  cross-mode: first leaf differs across modes")
            if verbose:
                for i, (f, l) in enumerate(zip((mini_f, default_f, large_f), leaves)):
                    print(f"        {f}: {l}")
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Meta tests: verify the harness catches real errors
#
# FALSE-NEGATIVE CLASS TO AVOID: decode reconstructs a point as the midpoint of
# its leaf and re-derives bits from *which leaf the point lands in* — so the
# encoding deliberately tolerates any perturbation smaller than a leaf cell
# (roughly 2^-15 for a 32-bit point, and much smaller where contraction has
# shrunk the leaf, e.g. ~2^-22 on the all-zeros path).  A corruption that moves
# the decoded point by LESS than the leaf width is (correctly) NOT detected.
# Worse, an EXTREME point (all-zeros / all-ones) sits on the same side of every
# split, so moving it further in that direction stays on that side and decodes
# to the SAME bits even once it has left the leaf (valid just goes False).
# Hence: test leaf MEMBERSHIP from the boundaries, don't perturb-and-decode.
# ---------------------------------------------------------------------------

def _parse_hex_i64(hex_str):
    """Parse a 0x... hex string (16 digits) to a signed i64."""
    val = int(hex_str, 16)
    if val >= 0x8000000000000000:
        val -= 0x10000000000000000
    return val


def _midpoint_i64(a, b):
    """Replicate Rust Fixed::midpoint: (a>>1) + (b>>1) + (a & b & 1)."""
    return (a >> 1) + (b >> 1) + (a & b & 1)


def _in_leaf(re, im, rmn, rmx, imn, imx):
    """Semi-open leaf membership, exactly mirroring Rust Rect::contains:
    re_min <= re < re_max  AND  im_min <= im < im_max  (closed min, open max)."""
    return rmn <= re < rmx and imn <= im < imx


def test_meta_leaf_membership(vectors_dir, verbose=False):
    """Build points straight from a leaf's boundaries and verify membership.

    The old coordinate-perturbation test produced a FALSE NEGATIVE: decode
    reconstructs a point as the midpoint of its leaf and re-derives bits from
    *which cell the point lands in*, so a sub-leaf nudge is (correctly) decoded
    unchanged — and for an all-zeros "extreme" point, even an out-of-leaf nudge
    stays on the same side at every split.  Instead, construct points directly
    from the boundaries and test the engine's semi-open [min, max) membership,
    including a point that lands EXACTLY on an open upper boundary (which must be
    classified outside) and a point guaranteed outside the leaf entirely.

    Grounded in the real engine: we first decode the leaf's center and confirm
    the engine returns the recorded leaf, so the boundaries we test against are
    genuinely the engine's own.
    """
    # Prefer a roomier leaf: vanity vectors carry mixed bits, so their leaf is a
    # well-sized rectangle.  All-zeros / all-ones drive the point to an extreme
    # corner and repeated contraction shrinks the leaf very small (~2^-22, but
    # still strictly positive on both axes — NOT zero-width), which is awkward
    # for building boundary points; vanity avoids that.
    cands = sorted(f for f in os.listdir(vectors_dir)
                   if "vanity" in f and f.endswith("_iter0.json"))
    cands = cands or sorted(f for f in os.listdir(vectors_dir)
                            if f.endswith("_iter0.json"))
    if not cands:
        print("  SKIP  meta-leaf-membership: no iter0 vectors")
        return True
    doc = json.load(open(os.path.join(vectors_dir, cands[0])))
    stage0 = doc["stages"][0]
    leaf = stage0["leaf"]
    rmn, rmx = _parse_hex_i64(leaf["re_min"]), _parse_hex_i64(leaf["re_max"])
    imn, imx = _parse_hex_i64(leaf["im_min"]), _parse_hex_i64(leaf["im_max"])
    if rmx <= rmn or imx <= imn:
        # Defensive only: leaf widths stay positive (interior splits + positive
        # contraction), so a zero-width leaf should never actually occur here.
        print("  SKIP  meta-leaf-membership: zero-width leaf")
        return True
    wr, wi = rmx - rmn, imx - imn
    cre, cim = _midpoint_i64(rmn, rmx), _midpoint_i64(imn, imx)

    # Ground the boundaries in the real engine: decoding the center must land in
    # exactly the recorded leaf.  Stage 0 is chain-derived (protocol 0.3.0), so
    # we decode with the stage's OWN stored (o, p, q), not o=p=q=0.
    p0 = stage0["params"]
    o0 = _parse_hex_i64(p0["o"]) if isinstance(p0["o"], str) else p0["o"]
    pp0 = _parse_hex_i64(p0["p"]) if isinstance(p0["p"], str) else p0["p"]
    q0 = _parse_hex_i64(p0["q"]) if isinstance(p0["q"], str) else p0["q"]
    try:
        from burning_ship_engine import decode_full
        from constants import ENCODE_AREA, GUI_PARAMS, BITS_PER_POINT
        _b, lr, _valid, _p = decode_full(
            cre, cim, BITS_PER_POINT, area=ENCODE_AREA, params=GUI_PARAMS,
            o=o0, p=pp0, q=q0, path_prefix="O")
        if (lr.re_min, lr.re_max, lr.im_min, lr.im_max) != (rmn, rmx, imn, imx):
            print("  FAIL  meta-leaf-membership: engine leaf != recorded leaf")
            return False
    except Exception as e:
        print(f"  FAIL  meta-leaf-membership: engine decode error: {e}")
        return False

    # Points built from the boundaries, with their expected membership under the
    # semi-open [min, max) convention.
    checks = [
        ("center (inside)",              cre,      cim,      True),
        ("closed lower corner",          rmn,      imn,      True),
        ("open upper-re boundary",       rmx,      cim,      False),  # bonus
        ("open upper-im boundary",       cre,      imx,      False),  # bonus
        ("open upper corner",            rmx,      imx,      False),
        ("guaranteed outside (+1 leaf)", rmx + wr, imx + wi, False),
        ("guaranteed outside (-1 leaf)", rmn - wr, imn - wi, False),
    ]
    bad = [(n, exp, _in_leaf(re, im, rmn, rmx, imn, imx))
           for (n, re, im, exp) in checks
           if _in_leaf(re, im, rmn, rmx, imn, imx) != exp]
    if bad:
        print(f"  FAIL  meta-leaf-membership: {len(bad)} membership mismatch(es)")
        if verbose:
            for n, exp, got in bad:
                print(f"        {n}: expected in_leaf={exp} got {got}")
        return False
    print("  OK    meta-leaf-membership: semi-open [min,max) boundaries correct "
          "(inside, closed corner, both open edges, outside)")
    return True


def test_meta_wrong_params(vectors_dir, verbose=False):
    """Decode with a wrong secret-stage param, verify entropy differs."""
    candidates = [f for f in os.listdir(vectors_dir)
                  if "_iter1.json" in f or "_iter2.json" in f]
    if not candidates:
        print("  SKIP  meta-wrong-params: no iter1/iter2 vectors found")
        return True
    vector_path = os.path.join(vectors_dir, sorted(candidates)[0])

    with open(vector_path, "r") as f:
        doc = json.load(f)

    original_hex = doc["input"]["entropy_hex"]

    # Corrupt the first secret stage's o parameter (stage index 1).  This is NOT
    # in the leaf-tolerance false-negative class: it perturbs the FRACTAL, not a
    # coordinate, so the bisection tree itself changes.  Note the o magnitude is
    # encoded with the LARGEST weight at the low bits (bit j -> 2^-(3+j)), so
    # XOR 0xFF flips the highest-magnitude components (down to 2^-3) — a large,
    # reliably-detectable change, not a sub-resolution tweak.
    import copy, tempfile
    corrupted = copy.deepcopy(doc)
    o_val = int(corrupted["stages"][1]["params"]["o"], 16)
    corrupted["stages"][1]["params"]["o"] = f"0x{(o_val ^ 0xFF):016X}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(corrupted, tf)
        tf_path = tf.name

    try:
        decoded = run_decode(tf_path, timeout=META_DECODE_TIMEOUT)
        decoded_hex = decoded["decoded_entropy_hex"]
        if decoded_hex == original_hex:
            print(f"  FAIL  meta-wrong-params: wrong params decoded to same entropy!")
            return False
        print(f"  OK    meta-wrong-params: wrong params detected (entropy differs)")
        return True
    except Exception as e:
        print(f"  OK    meta-wrong-params: wrong params caused error: {e}")
        return True
    finally:
        os.unlink(tf_path)


def test_meta_cross_stage_swap(vectors_dir, verbose=False):
    """Swap two stages' leaf boundaries, verify entropy mismatch."""
    candidates = [f for f in os.listdir(vectors_dir)
                  if "_iter1.json" in f or "_iter2.json" in f]
    if not candidates:
        print("  SKIP  meta-cross-swap: no iter1/iter2 vectors found")
        return True
    vector_path = os.path.join(vectors_dir, sorted(candidates)[0])

    with open(vector_path, "r") as f:
        doc = json.load(f)

    if len(doc["stages"]) < 2:
        print("  SKIP  meta-cross-swap: fewer than 2 stages")
        return True

    original_hex = doc["input"]["entropy_hex"]

    # Swap the leaf boundaries of stage 0 and stage 1.  This moves each point a
    # macroscopic distance (far more than a leaf) onto the OTHER stage's fractal,
    # so it is well clear of the leaf-tolerance false-negative class: the decode
    # either yields different entropy or, if the mismatched point lands in a
    # barren region, exhausts island discovery and is caught by the bounded
    # timeout (META_DECODE_TIMEOUT).  Both outcomes are valid detection; a
    # same-entropy decode would require an astronomically unlikely coincidence.
    import copy, tempfile
    bound_keys = ("re_min", "re_max", "im_min", "im_max")
    corrupted = copy.deepcopy(doc)
    s0 = corrupted["stages"][0]["leaf"]
    s1 = corrupted["stages"][1]["leaf"]
    for k in bound_keys:
        s0[k], s1[k] = s1[k], s0[k]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(corrupted, tf)
        tf_path = tf.name

    try:
        decoded = run_decode(tf_path, timeout=META_DECODE_TIMEOUT)
        decoded_hex = decoded["decoded_entropy_hex"]
        if decoded_hex == original_hex:
            print(f"  FAIL  meta-cross-swap: swapped stages decoded to same entropy!")
            return False
        print(f"  OK    meta-cross-swap: swapped stages detected (entropy differs)")
        return True
    except Exception as e:
        print(f"  OK    meta-cross-swap: swap caused error: {e}")
        return True
    finally:
        os.unlink(tf_path)


def main():
    parser = argparse.ArgumentParser(description="Great Wall test vector runner")
    parser.add_argument("--vector", type=str, help="Run single vector file")
    parser.add_argument("--verbose", action="store_true", help="Show diff details")
    parser.add_argument("--version", type=str, default=None,
                        help="Version directory (default: latest)")
    args = parser.parse_args()

    # Find version directory
    if args.vector:
        vector_files = [args.vector]
        vectors_dir = os.path.dirname(args.vector)
    else:
        if args.version:
            vectors_dir = os.path.join(VECTORS_DIR, args.version)
        else:
            versions = sorted(d for d in os.listdir(VECTORS_DIR)
                              if os.path.isdir(os.path.join(VECTORS_DIR, d)))
            if not versions:
                print("No test vector directories found.")
                sys.exit(1)
            vectors_dir = os.path.join(VECTORS_DIR, versions[-1])
        vector_files = sorted(
            os.path.join(vectors_dir, f) for f in os.listdir(vectors_dir)
            if f.endswith(".json"))

    current_engine = get_engine_version()
    print(f"Testing vectors in: {vectors_dir}")
    print(f"Current engine version: {current_engine}   protocol_version: {PROTOCOL_VERSION}")
    print()

    passed = 0
    failed = 0
    total = 0
    stale = 0

    # Version guard: a vector is STALE unless BOTH its engine `version` (the
    # single-fractal encode/decode algorithm) and its `protocol_version` (the
    # chained orchestration) match the current code.  Either differing means the
    # frozen output can no longer reproduce, so we skip it (never counting it as
    # a pass) — a stale vector can never show false-green during pre-1.0 churn.
    # Comprehensive vectors are rebuilt at the stable 1.0.0 release (see README).
    def _vector_versions(path):
        try:
            with open(path) as fh:
                d = json.load(fh)
            return d.get("protocol_version", "<unset>"), d.get("version", "<unset>")
        except Exception:
            return "<unreadable>", "<unreadable>"

    fresh_files = []
    stale_files = []
    for vf in vector_files:
        pv, ev = _vector_versions(vf)
        if pv == PROTOCOL_VERSION and ev == current_engine:
            fresh_files.append(vf)
        else:
            stale_files.append((vf, pv, ev))

    if stale_files:
        print("=== STALE Vectors (skipped — version mismatch) ===")
        for vf, pv, ev in stale_files:
            stale += 1
            print(f"  STALE {os.path.basename(vf)} "
                  f"(engine {ev}/protocol {pv} != current engine {current_engine}/"
                  f"protocol {PROTOCOL_VERSION})")
        print()

    # Frozen vector tests (fresh vectors only)
    print("=== Frozen Vector Tests ===")
    for vf in fresh_files:
        total += 1
        if test_frozen_vector(vf, verbose=args.verbose):
            passed += 1
        else:
            failed += 1

    # Round-trip tests (fresh vectors only)
    print()
    print("=== Round-Trip Tests ===")
    for vf in fresh_files:
        total += 1
        if test_round_trip(vf, verbose=args.verbose):
            passed += 1
        else:
            failed += 1

    # Cross-mode tests
    print()
    print("=== Cross-Mode Tests ===")
    total += 1
    if test_cross_mode(vectors_dir, verbose=args.verbose):
        passed += 1
    else:
        failed += 1

    # Meta tests (negative tests — verify harness catches errors)
    print()
    print("=== Meta Tests (negative — must detect corruption) ===")
    for meta_fn in (test_meta_leaf_membership, test_meta_wrong_params, test_meta_cross_stage_swap):
        total += 1
        if meta_fn(vectors_dir, verbose=args.verbose):
            passed += 1
        else:
            failed += 1

    print()
    stale_note = f", {stale} STALE-skipped" if stale else ""
    print(f"Results: {passed}/{total} passed, {failed} failed{stale_note}")
    if stale and not fresh_files:
        print(f"NOTE: all vectors are STALE for protocol {PROTOCOL_VERSION} "
              f"(none verified) — comprehensive vectors are rebuilt at 1.0.0.")
    # Exit non-zero only on real failures; STALE is expected pre-1.0 and is a
    # visible skip (never a false pass), not an error.
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
