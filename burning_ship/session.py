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
    PROFILE_BASIC,
)
from encoding import compute_checksum_bits
import protocol


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
    """Save the current encoding session to a JSON file (F6).

    Serialized with the chained, one-point-per-stage layout: per-stage leaf
    centers, per-stage params (list of dicts; element 0 is null = canonical),
    the active stage index, and the cumulative path/marker.
    """
    entropy_bits = []
    for chunks in state.stages_encoded_bits_chunks:
        for c in chunks:
            entropy_bits.extend(c)

    expected_ent = state.entropy_bits
    checksum_bits = compute_checksum_bits(entropy_bits) if len(entropy_bits) == expected_ent else []
    mnemonic = bits_to_mnemonic(entropy_bits) if len(entropy_bits) == expected_ent else ""

    # Per-stage leaf-area centers (one point per stage).
    stages_centers = []
    for pts in state.stages_encoded_points:
        centers = []
        for (re, im, re_raw, im_raw) in pts:
            centers.append({"re_raw": re_raw, "im_raw": im_raw,
                            "re_f64": re, "im_f64": im})
        stages_centers.append(centers)

    doc = {
        "stages_leaf_centers": stages_centers,
        "stages_params": state.stages_params,
        "stage": state.stage,
        "hashing": {
            "profile": state.argon2_profile,
            "iterations": int(state.argon2_iterations) if state.argon2_iterations else 0,
        },
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
    """Load an encoding session from a JSON file (F5).

    Detects the preset from the entropy length, then rebuilds every per-stage
    list by re-driving ``protocol.encode_entropy`` (the chained pipeline) so the
    stored points/params are reproduced deterministically.
    """
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

    # Hashing parameters
    hashing = doc.get("hashing", {})
    state.argon2_profile = hashing.get("profile", PROFILE_BASIC)
    iters = hashing.get("iterations", 0)
    state.argon2_iterations = str(iters)
    state.argon2_iter_cursor = len(state.argon2_iterations)

    # Digests / path info
    digests = doc.get("digests", {})
    state.argon2_digest = digests.get("argon2", "")
    state.stage1_path = doc.get("stage1_path", "O")
    state.argon2_marker = doc.get("argon2_marker", "")

    # Rebuild all per-stage lists sized to the detected preset, then re-encode
    # the full chain (N-1 derivations) from the stored entropy bits.
    state.reset_stage_data()
    from viewer import populate_stages_from_results  # GUI helper; avoids cycle
    stages = protocol.encode_entropy(entropy_bits, state.argon2_profile, iters)
    populate_stages_from_results(state, stages)
    state.argon2_stage1_bits = list(entropy_bits)

    # Restore the active stage if it is valid, else default to 0.
    saved_stage = doc.get("stage", 0)
    if isinstance(saved_stage, int) and 0 <= saved_stage < state.n_stages:
        state.stage = saved_stage
    else:
        state.stage = 0

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
