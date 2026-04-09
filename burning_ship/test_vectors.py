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
    """Test that the first point's leaf is identical across mini/default/large."""
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
            leaf = doc["stage1"]["leaves"][0]
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
    corrupted["stage1"]["leaves"][0]["re_min"] = _flip_hex_bit(
        corrupted["stage1"]["leaves"][0]["re_min"], 30)

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
    """Decode with wrong stage-2 params, verify entropy differs."""
    candidates = [f for f in os.listdir(vectors_dir)
                  if "_iter1.json" in f or "_iter2.json" in f]
    if not candidates:
        print("  SKIP  meta-wrong-params: no iter1/iter2 vectors found")
        return True
    vector_path = os.path.join(vectors_dir, sorted(candidates)[0])

    with open(vector_path, "r") as f:
        doc = json.load(f)

    original_hex = doc["input"]["entropy_hex"]

    # Corrupt stage-2 o parameter
    import copy, tempfile
    corrupted = copy.deepcopy(doc)
    o_val = int(corrupted["stage2"]["params"]["o"], 16)
    corrupted["stage2"]["params"]["o"] = f"0x{(o_val ^ 0xFF):016X}"

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
    """Swap stage1/stage2 leaf centers, verify entropy mismatch."""
    candidates = [f for f in os.listdir(vectors_dir)
                  if "_iter1.json" in f or "_iter2.json" in f]
    if not candidates:
        print("  SKIP  meta-cross-swap: no iter1/iter2 vectors found")
        return True
    vector_path = os.path.join(vectors_dir, sorted(candidates)[0])

    with open(vector_path, "r") as f:
        doc = json.load(f)

    if len(doc["stage1"]["leaves"]) < 1 or len(doc["stage2"]["leaves"]) < 1:
        print("  SKIP  meta-cross-swap: not enough leaves")
        return True

    original_hex = doc["input"]["entropy_hex"]

    # Swap first leaf of stage1 with first leaf of stage2
    import copy, tempfile
    corrupted = copy.deepcopy(doc)
    s1_center = (corrupted["stage1"]["leaves"][0]["center_re"],
                 corrupted["stage1"]["leaves"][0]["center_im"])
    s2_center = (corrupted["stage2"]["leaves"][0]["center_re"],
                 corrupted["stage2"]["leaves"][0]["center_im"])
    corrupted["stage1"]["leaves"][0]["center_re"] = s2_center[0]
    corrupted["stage1"]["leaves"][0]["center_im"] = s2_center[1]
    corrupted["stage2"]["leaves"][0]["center_re"] = s1_center[0]
    corrupted["stage2"]["leaves"][0]["center_im"] = s1_center[1]

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
    print()

    passed = 0
    failed = 0
    total = 0

    # Frozen vector tests
    print("=== Frozen Vector Tests ===")
    for vf in vector_files:
        total += 1
        if test_frozen_vector(vf, verbose=args.verbose):
            passed += 1
        else:
            failed += 1

    # Round-trip tests
    print()
    print("=== Round-Trip Tests ===")
    for vf in vector_files:
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
    print(f"Results: {passed}/{total} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
