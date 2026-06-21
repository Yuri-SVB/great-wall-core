#!/usr/bin/env python3
"""
Great Wall CLI — deterministic encode/decode with JSON output.

Protocol 0.3.0: a mandatory, point-less STAGE 0 carries only a short text input
(``--stage0-text``); the protocol then encodes one 32-bit point per LATER stage.
Every point stage's fractal is derived by hashing stage-0 text plus all
preceding points through the memory-hard chain (Argon2 → SHA-256 → (o,p,q)) —
there is no canonical fractal.  ``n_stages`` (= N point stages) = entropy_bits/32.

Usage:
  # Encode from hex entropy (stage-0 text seeds the chain)
  python3 cli.py encode --entropy a1b2c3d4... --stage0-text MAIN-STASH \
      --profile b --iterations 3 --mode d

  # Encode from BIP39 mnemonic, also exporting a master secret at the last stage
  python3 cli.py encode --bip39 "abandon abandon ..." --stage0-text RETIREMENT \
      --profile b --iterations 3 --mode d --export-label SIGNING-1

  # Decode from a stage document JSON
  python3 cli.py decode --input vectors.json

  # Master-secret export from an existing encode document
  python3 cli.py master --input vectors.json --export-label 2026-06-19
"""

import sys
import os
import json
import argparse

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
    SIZE_PRESETS, ARGON2_INPUT_BYTES,
)
from encoding import bits_to_bytes, bits_to_hex, stage_text_bytes, normalize_stage_text
from protocol import (
    encode_entropy, PROTOCOL_VERSION,
    export_master_secret, master_secret_display, _midpoint as _proto_midpoint,
)

# ---------------------------------------------------------------------------
# Profile mapping
# ---------------------------------------------------------------------------

PROFILE_MAP = {
    "b": PROFILE_BASIC,
    "a": PROFILE_ADVANCED,
    "g": PROFILE_GREAT_WALL,
}
PROFILE_NAMES = {"b": "basic", "a": "advanced", "g": "great_wall"}

# Back-compat short modes (m/d/l) map onto the word-count preset keys.
MODE_MAP = {
    "m": "6w",
    "d": "12w",
    "l": "24w",
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


def _params_dict(o, p, q, params_tuple):
    """Build the JSON params dict for one point stage's fractal.

    ``params_tuple`` is the 9-tuple from the chain derivation (with float
    display values).  Under protocol 0.3.0 every point stage is derived, so this
    is always present.
    """
    (_o, o_re, o_im, _p, p_re, p_im, _q, q_re, q_im) = params_tuple
    return {
        "o": _hex_fixed(o),
        "p": _hex_fixed(p),
        "q": _hex_fixed(q),
        "o_re": o_re, "o_im": o_im,
        "p_re": p_re, "p_im": p_im,
        "q_re": q_re, "q_im": q_im,
    }


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
    # Resolve the size preset.  --words selects any supported size (3..24, a
    # multiple of 3); --mode m/d/l is the legacy 6/12/24-word shortcut.
    if args.words is not None:
        preset_key = f"{args.words}w"
        if preset_key not in SIZE_PRESETS:
            valid = ", ".join(str(p["bip39_words"]) for p in SIZE_PRESETS.values())
            print(f"Error: unsupported --words {args.words} (choose from {valid})",
                  file=sys.stderr)
            sys.exit(1)
    elif args.mode is not None:
        preset_key = MODE_MAP.get(args.mode)
        if preset_key is None:
            print(f"Error: unknown mode '{args.mode}' (use m/d/l)", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: provide --words N or --mode m/d/l", file=sys.stderr)
        sys.exit(1)
    preset = SIZE_PRESETS[preset_key]
    total_entropy = preset["entropy_bits"]
    n_stages = preset["n_stages"]

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

    # Stage-0 text seeds the chain (normalized to [A-Z0-9-]; may be empty).
    stage0_text, _changed = normalize_stage_text(args.stage0_text or "")

    # Run the chained pipeline: one point per (later) stage, fractal k derived
    # from the memory-hard hash of stage-0 text + all preceding points.
    stages = encode_entropy(entropy_bits, stage0_text, profile_id, iterations)

    mnemonic = bits_to_mnemonic(entropy_bits)

    # Build per-stage output records (every point stage is chain-derived).
    stage0_prefix = stage_text_bytes(stage0_text)
    stage_docs = []
    for s in stages:
        leaf = _rect_to_dict(s.result.final_rect)
        leaf["path"] = s.result.path
        prior_bits = entropy_bits[:s.index * BITS_PER_POINT]
        argon2 = {
            "input_hex": (stage0_prefix + bits_to_bytes(prior_bits)).hex(),
            "iterations": iterations,
            "final_digest": s.digest.hex(),
        }
        stage_docs.append({
            "index": s.index,
            "params": _params_dict(s.o, s.p, s.q, s.params),
            "argon2": argon2,
            "leaf": leaf,
        })

    doc = {
        "version": get_engine_version(),
        "protocol_version": PROTOCOL_VERSION,
        "input": {
            "entropy_hex": bits_to_hex(entropy_bits),
            "entropy_bits": total_entropy,
            "bip39_mnemonic": mnemonic,
            "stage0_text": stage0_text,
            "argon2_profile": args.profile,
            "argon2_iterations": iterations,
            "gw_mode": args.mode,
            "bip39_words": preset["bip39_words"],
            "n_stages": n_stages,
        },
        "stages": stage_docs,
    }

    # Optional master-secret export at a chosen stage (DESIGN.md §Master-Secret
    # Export).  Available at any point stage; defaults to the last.
    if args.export_label is not None:
        export_idx = (args.export_stage - 1 if args.export_stage is not None
                      else len(stages) - 1)
        if not (0 <= export_idx < len(stages)):
            print(f"Error: --export-stage must be 1..{len(stages)}", file=sys.stderr)
            sys.exit(1)
        export_label, _c = normalize_stage_text(args.export_label)
        records = []
        for sr in stages[:export_idx + 1]:
            leaf_re = _proto_midpoint(sr.result.final_rect.re_min,
                                      sr.result.final_rect.re_max)
            leaf_im = _proto_midpoint(sr.result.final_rect.im_min,
                                      sr.result.final_rect.im_max)
            records.append((sr.o, sr.p, sr.q, leaf_re, leaf_im))
        raw = export_master_secret(stage0_text, iterations, records, export_label)
        doc["master_secret"] = {
            "export_stage": export_idx + 1,
            "export_label": export_label,
            "display": master_secret_display(raw),
            "full_hex": raw.hex(),
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
    protocol_version = doc.get("protocol_version", "unknown")
    stages = doc["stages"]

    # Decode each stage's point using its stored fractal parameters.  (The
    # document carries the per-stage params so a vector can be checked without
    # re-running the memory-hard chain; the live protocol re-derives them.)
    all_bits = []
    for st in sorted(stages, key=lambda s: s["index"]):
        params = st["params"]
        o = params["o"] if not isinstance(params["o"], str) else _parse_hex_i64(params["o"])
        p = params["p"] if not isinstance(params["p"], str) else _parse_hex_i64(params["p"])
        q = params["q"] if not isinstance(params["q"], str) else _parse_hex_i64(params["q"])
        re_raw, im_raw = _center_from_leaf(st["leaf"])
        bits, _rect, _valid, _path = decode_full(
            re_raw, im_raw, BITS_PER_POINT,
            area=ENCODE_AREA, params=GUI_PARAMS,
            o=o, p=p, q=q, path_prefix="O")
        all_bits.extend(bits)

    mnemonic = bits_to_mnemonic(all_bits)

    result = {
        "version": version,
        "protocol_version": protocol_version,
        "decoded_entropy_hex": bits_to_hex(all_bits),
        "decoded_entropy_bits": len(all_bits),
        "bip39_mnemonic": mnemonic,
        "n_stages": len(stages),
    }

    json.dump(result, sys.stdout, indent=2)
    print()


# ---------------------------------------------------------------------------
# Master-secret export command
# ---------------------------------------------------------------------------

def cmd_master(args):
    """Export a master secret from an existing encode document."""
    with open(args.input, "r") as f:
        doc = json.load(f)

    inp = doc.get("input", {})
    stage0_text, _c = normalize_stage_text(inp.get("stage0_text", ""))
    iterations = int(inp.get("argon2_iterations", 0))
    stages = sorted(doc["stages"], key=lambda s: s["index"])

    export_idx = (args.export_stage - 1 if args.export_stage is not None
                  else len(stages) - 1)
    if not (0 <= export_idx < len(stages)):
        print(f"Error: --export-stage must be 1..{len(stages)}", file=sys.stderr)
        sys.exit(1)
    export_label, _c2 = normalize_stage_text(args.export_label)

    records = []
    for st in stages[:export_idx + 1]:
        params = st["params"]
        o = _parse_hex_i64(params["o"]) if isinstance(params["o"], str) else params["o"]
        p = _parse_hex_i64(params["p"]) if isinstance(params["p"], str) else params["p"]
        q = _parse_hex_i64(params["q"]) if isinstance(params["q"], str) else params["q"]
        # uint64 transcript fields: re-wrap signed-parsed values to unsigned.
        o &= 0xFFFFFFFFFFFFFFFF
        p &= 0xFFFFFFFFFFFFFFFF
        q &= 0xFFFFFFFFFFFFFFFF
        leaf_re, leaf_im = _center_from_leaf(st["leaf"])
        records.append((o, p, q, leaf_re, leaf_im))

    raw = export_master_secret(stage0_text, iterations, records, export_label)
    result = {
        "version": doc.get("version", "unknown"),
        "protocol_version": doc.get("protocol_version", "unknown"),
        "export_stage": export_idx + 1,
        "export_label": export_label,
        "display": master_secret_display(raw),
        "full_hex": raw.hex(),
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
    enc = sub.add_parser("encode", help="Encode entropy → fractal stages (JSON)")
    enc.add_argument("--entropy", type=str, help="Hex string of entropy bits")
    enc.add_argument("--bip39", type=str, help="BIP39 mnemonic (alternative to --entropy)")
    enc.add_argument("--profile", type=str, required=True,
                     help="Argon2 profile: b=basic, a=advanced, g=great_wall")
    enc.add_argument("--iterations", type=int, required=True,
                     help="Argon2 iterations per stage link (0=identity); the "
                          "same count is applied to every stage")
    enc.add_argument("--words", type=int, default=None,
                     help="Mnemonic size in words: any multiple of 3 from 3 to "
                          "24 (= 32..256 entropy bits = 1..8 stages)")
    enc.add_argument("--mode", type=str, default=None,
                     help="Legacy size shortcut: m=6w, d=12w, l=24w "
                          "(use --words for other sizes)")
    enc.add_argument("--stage0-text", type=str, default="",
                     help="Mandatory stage-0 text that seeds the chain (salt/"
                          "pepper); normalized to [A-Z0-9-]. May be empty.")
    enc.add_argument("--export-label", type=str, default=None,
                     help="If set, also export a master secret using this label "
                          "(normalized to [A-Z0-9-]).")
    enc.add_argument("--export-stage", type=int, default=None,
                     help="Point stage (1..N) to export the master secret at "
                          "(default: the last stage).")

    # decode
    dec = sub.add_parser("decode", help="Decode a stage document JSON → entropy")
    dec.add_argument("--input", type=str, required=True,
                     help="Path to encode output JSON")

    # master
    mst = sub.add_parser("master", help="Export a master secret from an encode JSON")
    mst.add_argument("--input", type=str, required=True,
                     help="Path to encode output JSON")
    mst.add_argument("--export-label", type=str, required=True,
                     help="Export label (normalized to [A-Z0-9-]).")
    mst.add_argument("--export-stage", type=int, default=None,
                     help="Point stage (1..N) to export at (default: last).")

    args = parser.parse_args()
    if args.command == "encode":
        cmd_encode(args)
    elif args.command == "decode":
        cmd_decode(args)
    elif args.command == "master":
        cmd_master(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
