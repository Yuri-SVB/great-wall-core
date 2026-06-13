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
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"cli.py encode failed: {result.stderr}")
    return json.loads(result.stdout)


def run_decode(vector_path):
    """Run cli.py decode on a vector file."""
    cmd = [sys.executable, CLI_PATH, "decode", "--input", vector_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"cli.py decode failed: {result.stderr}")
    return json.loads(result.stdout)


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

    Stage 0 is the canonical fractal and its 32-bit point depends only on the
    first 32 entropy bits, which are identical for the all-zero "abandon"
    mnemonics across presets — so the first leaf must match.
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
# ---------------------------------------------------------------------------

def _flip_hex_bit(hex_str, bit_pos=0):
    """Flip one bit in a hex string (0x... format)."""
    val = int(hex_str, 16)
    val ^= (1 << bit_pos)
    return f"0x{val:016X}"


def test_meta_bitflip(vectors_dir, verbose=False):
    """Flip one bit in a leaf boundary and verify round-trip detects mismatch.

    Uses boundaries (re_min/re_max/im_min/im_max) rather than center,
    because the center is midpoint(min, max) and flipping its LSB might
    not change the midpoint due to rounding.
    """
    # Pick the first default vector available
    candidates = [f for f in os.listdir(vectors_dir)
                  if f.startswith("default_") and f.endswith("_iter0.json")]
    if not candidates:
        print("  SKIP  meta-bitflip: no default_*_iter0.json found")
        return True
    vector_path = os.path.join(vectors_dir, sorted(candidates)[0])

    with open(vector_path, "r") as f:
        doc = json.load(f)

    original_hex = doc["input"]["entropy_hex"]

    # Corrupt one leaf boundary (re_min, bit 30 — well inside the value)
    import copy
    corrupted = copy.deepcopy(doc)
    corrupted["stages"][0]["leaf"]["re_min"] = _flip_hex_bit(
        corrupted["stages"][0]["leaf"]["re_min"], 30)

    # Write to temp file and decode
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(corrupted, tf)
        tf_path = tf.name

    try:
        decoded = run_decode(tf_path)
        decoded_hex = decoded["decoded_entropy_hex"]
        if decoded_hex == original_hex:
            print(f"  FAIL  meta-bitflip: corrupted boundary decoded to same entropy!")
            return False
        print(f"  OK    meta-bitflip: corrupted boundary detected (entropy differs)")
        return True
    except Exception as e:
        # Decode error is also acceptable — corruption detected
        print(f"  OK    meta-bitflip: corruption caused decode error: {e}")
        return True
    finally:
        os.unlink(tf_path)


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

    # Corrupt the first secret stage's o parameter (stage index 1).
    import copy, tempfile
    corrupted = copy.deepcopy(doc)
    o_val = int(corrupted["stages"][1]["params"]["o"], 16)
    corrupted["stages"][1]["params"]["o"] = f"0x{(o_val ^ 0xFF):016X}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(corrupted, tf)
        tf_path = tf.name

    try:
        decoded = run_decode(tf_path)
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

    # Swap the leaf boundaries of stage 0 and stage 1 (different fractals →
    # different points → the decoded entropy must change).
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
        decoded = run_decode(tf_path)
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

    print(f"Testing vectors in: {vectors_dir}")
    print(f"Current protocol_version: {PROTOCOL_VERSION}")
    print()

    passed = 0
    failed = 0
    total = 0
    stale = 0

    # Version guard: a vector generated for a different protocol_version is
    # STALE.  We skip it (never counting it as a pass) so a stale vector can
    # never show false-green during pre-1.0 protocol churn.  Comprehensive
    # vectors are rebuilt at the stable 1.0.0 release (see DESIGN.md / README).
    def _vector_version(path):
        try:
            with open(path) as fh:
                return json.load(fh).get("protocol_version", "<unset>")
        except Exception:
            return "<unreadable>"

    fresh_files = []
    stale_files = []
    for vf in vector_files:
        if _vector_version(vf) == PROTOCOL_VERSION:
            fresh_files.append(vf)
        else:
            stale_files.append(vf)

    if stale_files:
        print("=== STALE Vectors (skipped — protocol_version mismatch) ===")
        for vf in stale_files:
            stale += 1
            print(f"  STALE {os.path.basename(vf)} "
                  f"(protocol {_vector_version(vf)} != current {PROTOCOL_VERSION})")
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
    for meta_fn in (test_meta_bitflip, test_meta_wrong_params, test_meta_cross_stage_swap):
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
