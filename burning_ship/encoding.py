"""
BIP39 ↔ fractal point encoding/decoding and bit-conversion utilities.
"""

import hashlib

from burning_ship_engine import encode, decode_full
from bip39 import mnemonic_to_bits, bits_to_mnemonic
from constants import (
    BITS_PER_POINT, ENCODE_AREA, GUI_PARAMS,
    STAGE1_O, STAGE1_P, STAGE1_Q,
    STAGE1_NUM_POINTS, STAGE2_NUM_POINTS,
    PROFILE_BASIC, PROFILE_ADVANCED, PROFILE_GREAT_WALL,
)


def argon2_path_marker(profile, iterations):
    """Build the Argon2 marker for the path string.

    Returns e.g. "B0", "A100", "G1000".
    """
    tag = {PROFILE_BASIC: "B", PROFILE_ADVANCED: "A", PROFILE_GREAT_WALL: "G"}.get(profile, "B")
    return f"{tag}{iterations}"


def encode_bip39(mnemonic_str, num_points=None):
    """Encode a BIP39 mnemonic into fractal points (first stage).

    num_points: points per stage (default: from global STAGE1_NUM_POINTS).
    """
    if num_points is None:
        num_points = STAGE1_NUM_POINTS
    bits = mnemonic_to_bits(mnemonic_str)
    entropy_bits = bits[:len(bits) - len(bits) // 33]  # strip checksum
    stage1_bits = entropy_bits[:num_points * BITS_PER_POINT]
    chunks = [stage1_bits[i*BITS_PER_POINT:(i+1)*BITS_PER_POINT]
              for i in range(num_points)]

    points = []
    all_steps = []
    final_rects = []
    for chunk in chunks:
        result = encode(chunk, area=ENCODE_AREA, params=GUI_PARAMS,
                        o=STAGE1_O, p=STAGE1_P, q=STAGE1_Q, path_prefix="O")
        points.append((
            result.point_re,
            result.point_im,
            result.point_re_raw,
            result.point_im_raw,
        ))
        all_steps.append(result.get_all_steps())
        final_rects.append(result.final_rect)
    return points, chunks, all_steps, final_rects


def encode_bip39_stage2(mnemonic_str, o, p, q, num_points=None):
    """Encode the last entropy bits into fractal points (stage 2)."""
    if num_points is None:
        num_points = STAGE2_NUM_POINTS
    bits = mnemonic_to_bits(mnemonic_str)
    entropy_bits = bits[:len(bits) - len(bits) // 33]  # strip checksum
    stage2_bits = entropy_bits[num_points * BITS_PER_POINT:]
    chunks = [stage2_bits[i*BITS_PER_POINT:(i+1)*BITS_PER_POINT]
              for i in range(num_points)]

    points = []
    all_steps = []
    final_rects = []
    for chunk in chunks:
        result = encode(chunk, area=ENCODE_AREA, params=GUI_PARAMS,
                        o=o, p=p, q=q, path_prefix="O")
        points.append((
            result.point_re,
            result.point_im,
            result.point_re_raw,
            result.point_im_raw,
        ))
        all_steps.append(result.get_all_steps())
        final_rects.append(result.final_rect)
    return points, chunks, all_steps, final_rects


def decode_points(raw_points, o=STAGE1_O, p=STAGE1_P, q=STAGE1_Q):
    """Decode raw (re_raw, im_raw) points back to entropy bits (32 per point).

    Returns (all_bits, step_lists, final_rects).
    """
    all_bits = []
    step_lists = []
    final_rects = []
    for re_raw, im_raw in raw_points:
        bits_chunk, leaf_rect, _valid, chunk_path = decode_full(
            re_raw, im_raw, BITS_PER_POINT, area=ENCODE_AREA,
            params=GUI_PARAMS, o=o, p=p, q=q, path_prefix="O")
        all_bits.extend(bits_chunk)
        final_rects.append(leaf_rect)
        result = encode(bits_chunk, area=ENCODE_AREA, params=GUI_PARAMS,
                        o=o, p=p, q=q, path_prefix="O")
        step_lists.append(result.get_all_steps())
    return all_bits, step_lists, final_rects


def encode_bits_stage(stage_bits, o, p, q):
    """Encode stage bits into fractal points (32 bits each).

    Returns (points, chunks, steps, final_rects).
    """
    num_points = len(stage_bits) // BITS_PER_POINT
    chunks = [stage_bits[i*BITS_PER_POINT:(i+1)*BITS_PER_POINT]
              for i in range(num_points)]
    points = []
    all_steps = []
    final_rects = []
    for chunk in chunks:
        result = encode(chunk, area=ENCODE_AREA, params=GUI_PARAMS,
                        o=o, p=p, q=q, path_prefix="O")
        points.append((result.point_re, result.point_im,
                        result.point_re_raw, result.point_im_raw))
        all_steps.append(result.get_all_steps())
        final_rects.append(result.final_rect)
    return points, chunks, all_steps, final_rects


# ---------------------------------------------------------------------------
# Bit-conversion utilities
# ---------------------------------------------------------------------------

def bits_to_bytes(bits):
    """Convert a list of 0/1 ints to a bytes object."""
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bits):
                byte_val = (byte_val << 1) | bits[i + j]
            else:
                byte_val <<= 1
        out.append(byte_val)
    return bytes(out)


def bits_to_hex(bits):
    """Convert a list of 0/1 ints to a hex string."""
    return bits_to_bytes(bits).hex()


def compute_checksum_bits(entropy_bits):
    """Compute the BIP39 checksum (len/32 bits) from entropy bits."""
    cs_len = len(entropy_bits) // 32
    entropy_bytes = bits_to_bytes(entropy_bits)
    sha = hashlib.sha256(entropy_bytes).digest()
    return [(sha[i // 8] >> (7 - i % 8)) & 1 for i in range(cs_len)]
