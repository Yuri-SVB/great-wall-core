"""
BIP39 ↔ fractal point encoding/decoding and bit-conversion utilities.
"""

import hashlib

from burning_ship_engine import encode, decode_full
from bip39 import mnemonic_to_bits, bits_to_mnemonic
from constants import (
    BITS_PER_POINT, ENCODE_AREA, GUI_PARAMS,
    STAGE1_O, STAGE1_P, STAGE1_Q,
    PROFILE_BASIC, PROFILE_ADVANCED, PROFILE_GREAT_WALL,
)


def argon2_path_marker(profile, iterations):
    """Build the Argon2 marker for the path string.

    Returns e.g. "B0", "A100", "G1000".
    """
    tag = {PROFILE_BASIC: "B", PROFILE_ADVANCED: "A", PROFILE_GREAT_WALL: "G"}.get(profile, "B")
    return f"{tag}{iterations}"


def encode_bip39(mnemonic_str, num_points=1):
    """Encode a BIP39 mnemonic into fractal points (first stage).

    DEPRECATED: legacy two-stage helper.  The one-point-per-stage protocol
    drives encoding through :func:`protocol.encode_entropy`; this remains only
    for the not-yet-migrated GUI path.  num_points defaults to 1 (one point
    per stage).
    """
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


def encode_bip39_stage2(mnemonic_str, o, p, q, num_points=1):
    """Encode the last entropy bits into fractal points (stage 2).

    DEPRECATED: legacy two-stage helper; see :func:`encode_bip39`.
    """
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


# ---------------------------------------------------------------------------
# Stage text normalization (protocol 0.3.0)
# ---------------------------------------------------------------------------
#
# Every stage text input — stage-0 text AND every non-0 stage's master-secret
# export label — is restricted to upper-case ASCII alphanumerics and '-' only
# ([A-Z0-9-]).  The restriction (DESIGN.md "Strong text restrictions") exists so
# the same text round-trips identically across devices, keyboards, locales, and
# clipboards: a stray lower-case letter, accent, or Unicode look-alike would
# silently fork the chain (or the export) into a different, unrecoverable result.

STAGE_TEXT_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")


def normalize_stage_text(text):
    """Normalize stage text to the ``[A-Z0-9-]`` set (DESIGN.md §Stage 0).

    Up-cases ASCII letters and drops every character outside ``[A-Z0-9-]``
    (accents, spaces, punctuation, Unicode look-alikes, control bytes).

    Returns ``(normalized, changed)`` where ``changed`` is True if any character
    was up-cased or dropped — so callers (e.g. the GUI field) can signal the user
    that a restriction was applied, and the divergence is never silent.
    """
    out = []
    changed = False
    for ch in text:
        up = ch.upper()
        if up != ch:
            changed = True
        if up in STAGE_TEXT_ALPHABET:
            out.append(up)
        else:
            changed = True
    return "".join(out), changed


def stage_text_bytes(text):
    """ASCII bytes of normalized stage text — the chain/export input form.

    Normalizes first (so callers may pass raw text) and encodes as ASCII; the
    ``[A-Z0-9-]`` restriction guarantees the encoding is always valid ASCII.
    """
    normalized, _changed = normalize_stage_text(text)
    return normalized.encode("ascii")


def compute_checksum_bits(entropy_bits):
    """Compute the BIP39 checksum (len/32 bits) from entropy bits."""
    cs_len = len(entropy_bits) // 32
    entropy_bytes = bits_to_bytes(entropy_bits)
    sha = hashlib.sha256(entropy_bytes).digest()
    return [(sha[i // 8] >> (7 - i % 8)) & 1 for i in range(cs_len)]
