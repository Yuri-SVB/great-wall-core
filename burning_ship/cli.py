#!/usr/bin/env python3
"""
Great Wall CLI — deterministic encode/decode with JSON output.

Usage:
  # Encode from hex entropy
  python3 cli.py encode --entropy a1b2c3d4... --profile b --iterations 3 --mode d

  # Encode from BIP39 mnemonic
  python3 cli.py encode --bip39 "abandon abandon ..." --profile b --iterations 3 --mode d

  # Decode from leaf centers JSON
  python3 cli.py decode --input vectors.json
"""

import sys
import os
import json
import argparse
import struct
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from burning_ship_engine import (
    encode, decode_full, get_engine_version, DiscoveryParams,
    PROFILE_BASIC, PROFILE_ADVANCED, PROFILE_GREAT_WALL,
    ARGON2_DIGEST_BYTES, argon2_single,
    fixed_to_f64,
)
from bip39 import mnemonic_to_bits, bits_to_mnemonic
from constants import (
    BITS_PER_POINT, ENCODE_AREA, GUI_PARAMS,
    STAGE1_O, STAGE1_P, STAGE1_Q,
    SIZE_PRESETS, ARGON2_INPUT_BYTES,
)
from encoding import bits_to_bytes, bits_to_hex
from argon2_pipeline import derive_stage2_params

# ---------------------------------------------------------------------------
# Profile mapping
# ---------------------------------------------------------------------------

PROFILE_MAP = {
    "b": PROFILE_BASIC,
    "a": PROFILE_ADVANCED,
    "g": PROFILE_GREAT_WALL,
}
PROFILE_NAMES = {"b": "basic", "a": "advanced", "g": "great_wall"}

MODE_MAP = {
    "m": "mini",
    "d": "default",
    "l": "large",
}


def _hex_fixed(raw_i64):
    """Format a raw Fixed i64 as a hex string (signed, 16 hex digits)."""
    return f"0x{raw_i64 & 0xFFFFFFFFFFFFFFFF:016X}"


def _rect_to_dict(rect):
    """Convert a Rect to a dict with hex Fixed bounds and f64 display values."""
    return {
        "re_min": _hex_fixed(rect.re_min),
        "re_max": _hex_fixed(rect.re_max),
        "im_min": _hex_fixed(rect.im_min),
        "im_max": _hex_fixed(rect.im_max),
        "re_min_f64": rect.re_min_f64(),
        "re_max_f64": rect.re_max_f64(),
        "im_min_f64": rect.im_min_f64(),
        "im_max_f64": rect.im_max_f64(),
    }


def _encode_stage(stage_bits, o, p, q):
    """Encode one stage's bits into points, returning leaf dicts."""
    num_points = len(stage_bits) // BITS_PER_POINT
    chunks = [stage_bits[i * BITS_PER_POINT:(i + 1) * BITS_PER_POINT]
              for i in range(num_points)]
    leaves = []
    for i, chunk in enumerate(chunks):
        result = encode(chunk, area=ENCODE_AREA, params=GUI_PARAMS,
                        o=o, p=p, q=q, path_prefix="O")
        leaf = _rect_to_dict(result.final_rect)
        leaf["point"] = i + 1
        leaf["path"] = result.path
        leaves.append(leaf)
    return leaves


def _run_argon2(stage1_bytes, profile_id, iterations):
    """Run iterative Argon2, collecting all intermediate digests."""
    digests = []
    if iterations == 0:
        digest = stage1_bytes.ljust(ARGON2_DIGEST_BYTES, b'\x00')[:ARGON2_DIGEST_BYTES]
        digests.append({"iteration": 0, "hex": digest.hex()})
    else:
        digest = argon2_single(stage1_bytes, profile_id)
        digests.append({"iteration": 1, "hex": digest.hex()})
        for i in range(1, iterations):
            digest = argon2_single(digest, profile_id)
            digests.append({"iteration": i + 1, "hex": digest.hex()})
    return digest, digests


def _parse_hex_i64(hex_str):
    """Parse a 0x... hex string to a signed i64."""
    val = int(hex_str, 16)
    if val >= 0x8000000000000000:
        val -= 0x10000000000000000
    return val


def _midpoint(a, b):
    """Replicate Rust's Fixed::midpoint: (a>>1) + (b>>1) + (a & b & 1)."""
    return (a >> 1) + (b >> 1) + (a & b & 1)


def _center_from_leaf(leaf):
    """Reconstruct the center point from leaf boundaries (canonical)."""
    re_min = _parse_hex_i64(leaf["re_min"])
    re_max = _parse_hex_i64(leaf["re_max"])
    im_min = _parse_hex_i64(leaf["im_min"])
    im_max = _parse_hex_i64(leaf["im_max"])
    return _midpoint(re_min, re_max), _midpoint(im_min, im_max)


def _decode_leaves(leaves, o, p, q):
    """Decode a list of leaf dicts to bits, using boundaries to derive centers."""
    all_bits = []
    for leaf in leaves:
        re_raw, im_raw = _center_from_leaf(leaf)
        bits, _rect, valid, _path = decode_full(
            re_raw, im_raw, BITS_PER_POINT,
            area=ENCODE_AREA, params=GUI_PARAMS,
            o=o, p=p, q=q, path_prefix="O")
        all_bits.extend(bits)
    return all_bits


def _entropy_from_hex(hex_str):
    """Convert hex string to list of 0/1 bits."""
    raw = bytes.fromhex(hex_str)
    bits = []
    for b in raw:
        for j in range(7, -1, -1):
            bits.append((b >> j) & 1)
    return bits


def _entropy_from_bip39(mnemonic_str):
    """Convert BIP39 mnemonic to entropy bits (strip checksum)."""
    all_bits = mnemonic_to_bits(mnemonic_str)
    # Strip checksum: total_bits - total_bits // 33
    entropy_len = len(all_bits) - len(all_bits) // 33
    return all_bits[:entropy_len]


# ---------------------------------------------------------------------------
# Encode command
# ---------------------------------------------------------------------------

def cmd_encode(args):
    # Parse mode
    mode_name = MODE_MAP.get(args.mode)
    if mode_name is None:
        print(f"Error: unknown mode '{args.mode}' (use m/d/l)", file=sys.stderr)
        sys.exit(1)
    preset = SIZE_PRESETS[mode_name]
    points_per_stage = preset["points_per_stage"]
    total_entropy = preset["entropy_bits"]
    bits_per_stage = points_per_stage * BITS_PER_POINT

    # Parse entropy
    if args.bip39:
        entropy_bits = _entropy_from_bip39(args.bip39)
    elif args.entropy:
        entropy_bits = _entropy_from_hex(args.entropy)
    else:
        print("Error: provide --entropy or --bip39", file=sys.stderr)
        sys.exit(1)

    if len(entropy_bits) != total_entropy:
        print(f"Error: expected {total_entropy} entropy bits for mode '{args.mode}', "
              f"got {len(entropy_bits)}", file=sys.stderr)
        sys.exit(1)

    # Parse Argon2 profile
    profile_id = PROFILE_MAP.get(args.profile)
    if profile_id is None:
        print(f"Error: unknown profile '{args.profile}' (use b/a/g)", file=sys.stderr)
        sys.exit(1)
    iterations = args.iterations

    # Split entropy
    stage1_bits = entropy_bits[:bits_per_stage]
    stage2_bits = entropy_bits[bits_per_stage:]

    # Stage 1
    stage1_leaves = _encode_stage(stage1_bits, STAGE1_O, STAGE1_P, STAGE1_Q)

    # Argon2
    stage1_bytes = bits_to_bytes(stage1_bits)
    final_digest, argon2_digests = _run_argon2(stage1_bytes, profile_id, iterations)

    # Derive stage-2 params
    o, o_re, o_im, p, p_re, p_im, q, q_re, q_im = derive_stage2_params(final_digest)

    # Stage 2
    stage2_leaves = _encode_stage(stage2_bits, o, p, q)

    # BIP39 mnemonic
    mnemonic = bits_to_mnemonic(entropy_bits)

    # Build output
    doc = {
        "version": get_engine_version(),
        "input": {
            "entropy_hex": bits_to_hex(entropy_bits),
            "entropy_bits": total_entropy,
            "bip39_mnemonic": mnemonic,
            "argon2_profile": args.profile,
            "argon2_iterations": iterations,
            "gw_mode": args.mode,
        },
        "stage1": {
            "params": {
                "o": STAGE1_O,
                "p": STAGE1_P,
                "q": STAGE1_Q,
            },
            "leaves": stage1_leaves,
        },
        "argon2": {
            "input_hex": stage1_bytes.hex(),
            "profile": PROFILE_NAMES[args.profile],
            "iterations": iterations,
            "digests": argon2_digests,
            "final_digest": final_digest.hex(),
        },
        "stage2": {
            "params": {
                "o": _hex_fixed(o),
                "p": _hex_fixed(p),
                "q": _hex_fixed(q),
                "o_re": o_re,
                "o_im": o_im,
                "p_re": p_re,
                "p_im": p_im,
                "q_re": q_re,
                "q_im": q_im,
            },
            "leaves": stage2_leaves,
        },
    }

    json.dump(doc, sys.stdout, indent=2)
    print()


# ---------------------------------------------------------------------------
# Decode command
# ---------------------------------------------------------------------------

def cmd_decode(args):
    with open(args.input, "r") as f:
        doc = json.load(f)

    version = doc.get("version", "unknown")

    # Stage-1 params
    s1_params = doc["stage1"]["params"]
    s1_o, s1_p, s1_q = s1_params["o"], s1_params["p"], s1_params["q"]

    # Decode stage 1 from leaf boundaries
    stage1_bits = _decode_leaves(doc["stage1"]["leaves"], s1_o, s1_p, s1_q)

    # Stage-2 params
    s2_params = doc["stage2"]["params"]
    s2_o = _parse_hex_i64(s2_params["o"]) if isinstance(s2_params["o"], str) else s2_params["o"]
    s2_p = _parse_hex_i64(s2_params["p"]) if isinstance(s2_params["p"], str) else s2_params["p"]
    s2_q = _parse_hex_i64(s2_params["q"]) if isinstance(s2_params["q"], str) else s2_params["q"]

    # Decode stage 2 from leaf boundaries
    stage2_bits = _decode_leaves(doc["stage2"]["leaves"], s2_o, s2_p, s2_q)

    all_entropy = stage1_bits + stage2_bits
    mnemonic = bits_to_mnemonic(all_entropy)

    result = {
        "version": version,
        "decoded_entropy_hex": bits_to_hex(all_entropy),
        "decoded_entropy_bits": len(all_entropy),
        "bip39_mnemonic": mnemonic,
        "stage1_hex": bits_to_hex(stage1_bits),
        "stage2_hex": bits_to_hex(stage2_bits),
    }

    json.dump(result, sys.stdout, indent=2)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Great Wall CLI — deterministic encode/decode with JSON output")
    sub = parser.add_subparsers(dest="command")

    # encode
    enc = sub.add_parser("encode", help="Encode entropy → fractal leaves (JSON)")
    enc.add_argument("--entropy", type=str, help="Hex string of entropy bits")
    enc.add_argument("--bip39", type=str, help="BIP39 mnemonic (alternative to --entropy)")
    enc.add_argument("--profile", type=str, required=True,
                     help="Argon2 profile: b=basic, a=advanced, g=great_wall")
    enc.add_argument("--iterations", type=int, required=True,
                     help="Number of Argon2 iterations (0=identity)")
    enc.add_argument("--mode", type=str, required=True,
                     help="Size mode: m=mini(6w), d=default(12w), l=large(24w)")

    # decode
    dec = sub.add_parser("decode", help="Decode leaf centers JSON → entropy")
    dec.add_argument("--input", type=str, required=True,
                     help="Path to encode output JSON")

    args = parser.parse_args()
    if args.command == "encode":
        cmd_encode(args)
    elif args.command == "decode":
        cmd_decode(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
