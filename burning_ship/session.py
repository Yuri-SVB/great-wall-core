"""
Session save/load (F5/F6) and clipboard helpers.
"""

import json
import subprocess

import pygame

from burning_ship_engine import cache_clear_stage2
from bip39 import bits_to_mnemonic
from constants import (
    SIZE_PRESETS,
    STAGE1_O, STAGE1_P, STAGE1_Q,
    PROFILE_BASIC,
)
from encoding import encode_bits_stage, compute_checksum_bits


# ---------------------------------------------------------------------------
# Clipboard helpers
# ---------------------------------------------------------------------------

def copy_to_clipboard(text):
    """Copy text to the system clipboard.

    Tries pygame.scrap first, then falls back to platform tools.
    Raises RuntimeError if all methods fail.
    """
    data = text.encode("utf-8")

    # Try pygame.scrap
    try:
        pygame.scrap.put(pygame.SCRAP_TEXT, data + b"\x00")
        return
    except Exception:
        pass

    # Try platform clipboard tools
    for cmd in (["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"],
                ["wl-copy"],
                ["pbcopy"]):
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc.communicate(input=data)
            if proc.returncode == 0:
                return
        except FileNotFoundError:
            continue

    raise RuntimeError("Install xclip, xsel, or wl-copy for clipboard support")


def paste_from_clipboard():
    """Read text from the system clipboard.

    Tries pygame.scrap first, then falls back to platform tools.
    Returns the text string, or None if all methods fail.
    """
    # Try pygame.scrap
    try:
        clip = pygame.scrap.get(pygame.SCRAP_TEXT)
        if clip:
            return clip.decode("utf-8", errors="replace").rstrip("\x00")
    except Exception:
        pass

    # Try platform clipboard tools
    for cmd in (["xclip", "-selection", "clipboard", "-o"],
                ["xsel", "--clipboard", "--output"],
                ["wl-paste", "--no-newline"],
                ["pbpaste"]):
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL)
            out, _ = proc.communicate()
            if proc.returncode == 0:
                return out.decode("utf-8", errors="replace")
        except FileNotFoundError:
            continue

    return None


# ---------------------------------------------------------------------------
# Session save / load (F6 / F5)
# ---------------------------------------------------------------------------

def save_session(state, path):
    """Save the current encoding session to a JSON file (F6)."""
    entropy_bits = []
    for c in state.stage1_encoded_bits_chunks:
        entropy_bits.extend(c)
    for c in state.stage2_encoded_bits_chunks:
        entropy_bits.extend(c)

    expected_ent = state.entropy_bits
    checksum_bits = compute_checksum_bits(entropy_bits) if len(entropy_bits) == expected_ent else []
    mnemonic = bits_to_mnemonic(entropy_bits) if len(entropy_bits) == expected_ent else ""

    # Leaf-area centers: fixed_to_f64 of each encoded point's raw coords
    s1_centers = []
    for (re, im, re_raw, im_raw) in state.stage1_encoded_points:
        s1_centers.append({"re_raw": re_raw, "im_raw": im_raw,
                           "re_f64": re, "im_f64": im})
    s2_centers = []
    for (re, im, re_raw, im_raw) in state.stage2_encoded_points:
        s2_centers.append({"re_raw": re_raw, "im_raw": im_raw,
                           "re_f64": re, "im_f64": im})

    doc = {
        "stage1_leaf_centers": s1_centers,
        "stage2_leaf_centers": s2_centers,
        "hashing": {
            "profile": state.argon2_profile,
            "iterations": int(state.argon2_iterations) if state.argon2_iterations else 0,
        },
        "stage2_o": state.stage2_o,
        "stage2_o_re": state.stage2_o_re,
        "stage2_o_im": state.stage2_o_im,
        "stage2_p": state.stage2_p,
        "stage2_p_re": state.stage2_p_re,
        "stage2_p_im": state.stage2_p_im,
        "stage2_q": state.stage2_q,
        "stage2_q_re": state.stage2_q_re,
        "stage2_q_im": state.stage2_q_im,
        "digests": {
            "argon2": state.argon2_digest,
        },
        "size_preset": state.size_preset,
        "entropy_bits": entropy_bits,
        "checksum_bits": checksum_bits,
        "bip39_mnemonic": mnemonic,
        "stage1_path": state.stage1_path,
        "argon2_marker": state.argon2_marker,
    }

    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def load_session(state, path):
    """Load an encoding session from a JSON file (F5)."""
    with open(path, "r") as f:
        doc = json.load(f)

    # Entropy bits — detect preset from length
    entropy_bits = doc.get("entropy_bits", [])
    n_ent = len(entropy_bits)
    matched_preset = None
    for name, cfg in SIZE_PRESETS.items():
        if cfg["entropy_bits"] == n_ent:
            matched_preset = name
            break
    if matched_preset is None:
        raise ValueError(f"Unsupported entropy length {n_ent} bits (expected 64, 128, or 256)")
    state.size_preset = matched_preset
    bps = state.bits_per_stage
    stage1_bits = entropy_bits[:bps]
    stage2_bits = entropy_bits[bps:]

    # Hashing parameters
    hashing = doc.get("hashing", {})
    state.argon2_profile = hashing.get("profile", PROFILE_BASIC)
    iters = hashing.get("iterations", 0)
    state.argon2_iterations = str(iters)
    state.argon2_iter_cursor = len(state.argon2_iterations)

    # Stage-2 parameters
    state.stage2_o = doc.get("stage2_o")
    state.stage2_o_re = doc.get("stage2_o_re")
    state.stage2_o_im = doc.get("stage2_o_im")
    state.stage2_p = doc.get("stage2_p")
    state.stage2_p_re = doc.get("stage2_p_re")
    state.stage2_p_im = doc.get("stage2_p_im")
    state.stage2_q = doc.get("stage2_q")
    state.stage2_q_re = doc.get("stage2_q_re")
    state.stage2_q_im = doc.get("stage2_q_im")

    # Digests
    digests = doc.get("digests", {})
    state.argon2_digest = digests.get("argon2", "")

    # Path info
    state.stage1_path = doc.get("stage1_path", "O")
    state.argon2_marker = doc.get("argon2_marker", "")

    # Re-encode stage 1 from bits to get points, steps, rects
    s1_pts, s1_chunks, s1_steps, s1_rects = \
        encode_bits_stage(stage1_bits, STAGE1_O, STAGE1_P, STAGE1_Q)
    state.stage1_encoded_points = s1_pts
    state.stage1_encoded_bits_chunks = s1_chunks
    state.stage1_encoded_steps = s1_steps
    state.stage1_encoded_final_rects = s1_rects
    state.argon2_stage1_bits = stage1_bits

    # Re-encode stage 2 from bits
    if state.stage2_p is not None and state.stage2_o is not None:
        s2_pts, s2_chunks, s2_steps, s2_rects = \
            encode_bits_stage(stage2_bits, state.stage2_o, state.stage2_p, state.stage2_q)
        state.stage2_encoded_points = s2_pts
        state.stage2_encoded_bits_chunks = s2_chunks
        state.stage2_encoded_steps = s2_steps
        state.stage2_encoded_final_rects = s2_rects
        state.stage = 2
    else:
        state.stage = 1

    # BIP39 mnemonic
    mnemonic = doc.get("bip39_mnemonic", "")
    if not mnemonic and len(entropy_bits) == state.entropy_bits:
        mnemonic = bits_to_mnemonic(entropy_bits)
    state.input_text = mnemonic
    state.input_cursor = len(mnemonic)
    state.input_sel = len(mnemonic)
    state.decoded_mnemonic = mnemonic

    state.selected_point_idx = None
    state.selected_decoded_idx = None
    cache_clear_stage2()
    state.needs_redraw = True
