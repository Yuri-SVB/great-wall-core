"""
Argon2 hashing pipeline, checkpoint management, stage-2 parameter
derivation, and F2 random-encode orchestration.
"""

import os
import struct
import hashlib
import threading

from burning_ship_engine import (
    argon2_single, cache_clear_stage2,
    PROFILE_BASIC, PROFILE_ADVANCED, PROFILE_GREAT_WALL,
    ARGON2_DIGEST_BYTES,
)
from bip39 import bits_to_mnemonic
from constants import (
    ARGON2_INPUT_BYTES,
    STAGE1_O, STAGE1_P, STAGE1_Q,
    P_MAGNITUDE_BITS, P_SIGN_BIT_RE, P_SIGN_BIT_IM,
    P_MAGNITUDE_MIN_EXP, P_BASELINE_EXP,
    Q_MAGNITUDE_BITS, Q_SIGN_BIT_RE, Q_SIGN_BIT_IM, Q_MAGNITUDE_MIN_EXP,
    O_MAGNITUDE_BITS, O_SIGN_BIT_RE, O_SIGN_BIT_IM, O_MAGNITUDE_MIN_EXP,
    CLR_PENDING, CLR_SUCCESS, CLR_ERROR,
)
from encoding import (
    argon2_path_marker, encode_bits_stage, bits_to_bytes,
)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def _checkpoint_path(input_hex, profile):
    """Return the checkpoint file path for a given input and profile."""
    _PROFILE_TAGS = {PROFILE_BASIC: "basic", PROFILE_ADVANCED: "advanced",
                     PROFILE_GREAT_WALL: "greatwall"}
    profile_tag = _PROFILE_TAGS.get(profile, "basic")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f".argon2_checkpoint_{input_hex}_{profile_tag}.bin")


def _save_checkpoint(path, iteration, digest):
    """Append one (iteration, digest) pair to the checkpoint file.

    File format: sequence of 36-byte records (4-byte LE iteration + 32-byte digest).
    """
    with open(path, "ab") as f:
        f.write(struct.pack('<I', iteration))
        f.write(digest)


def _load_checkpoint(path):
    """Load all checkpointed (iteration, digest) pairs.

    Returns a dict {iteration: digest_bytes}.
    """
    records = {}
    if not os.path.exists(path):
        return records
    with open(path, "rb") as f:
        while True:
            header = f.read(4)
            if len(header) < 4:
                break
            it = struct.unpack('<I', header)[0]
            digest = f.read(ARGON2_DIGEST_BYTES)
            if len(digest) < ARGON2_DIGEST_BYTES:
                break
            records[it] = digest
    return records


# ---------------------------------------------------------------------------
# Iterative Argon2 hashing (background thread)
# ---------------------------------------------------------------------------

def run_argon2_iterative(state, gui_iterations):
    """Run iterative Argon2d in a background thread, updating state.argon2_progress.

    Each iteration calls argon2_single() (one Argon2d pass via Rust) and
    feeds the 32-byte digest back as input for the next iteration.
    gui_iterations=0 means identity (no Argon2).

    Intermediate digests are checkpointed to disk for resumption.
    """
    stage1_bits = state.argon2_stage1_bits
    profile = state.argon2_profile

    def _worker():
        try:
            data = bits_to_bytes(stage1_bits)
            if gui_iterations == 0:
                digest = data.ljust(ARGON2_DIGEST_BYTES, b'\x00')[:ARGON2_DIGEST_BYTES]
                state.argon2_progress = 1
            else:
                input_hex = data.hex()
                ckpt_path = _checkpoint_path(input_hex, profile)
                saved = _load_checkpoint(ckpt_path)

                resume_it = 0
                digest = None
                for it in sorted(saved.keys()):
                    if it <= gui_iterations:
                        resume_it = it
                        digest = saved[it]

                if resume_it == 0:
                    digest = argon2_single(data, profile)
                    _save_checkpoint(ckpt_path, 1, digest)
                    state.argon2_progress = 1
                    resume_it = 1
                else:
                    state.argon2_progress = resume_it

                for i in range(resume_it, gui_iterations):
                    digest = argon2_single(digest, profile)
                    cur_it = i + 1
                    _save_checkpoint(ckpt_path, cur_it, digest)
                    state.argon2_progress = cur_it

            state.argon2_digest = digest.hex()
            state.stage2_o, state.stage2_o_re, state.stage2_o_im, \
                state.stage2_p, state.stage2_p_re, state.stage2_p_im, \
                state.stage2_q, state.stage2_q_re, state.stage2_q_im = derive_stage2_params(digest)
            state.argon2_marker = argon2_path_marker(profile, gui_iterations)
            cache_clear_stage2()
            state.stage = 2
            state.needs_redraw = True
            _PROFILE_LABELS = {PROFILE_BASIC: "Basic", PROFILE_ADVANCED: "Advanced",
                               PROFILE_GREAT_WALL: "Great Wall"}
            profile_label = _PROFILE_LABELS.get(profile, "Basic")
            label = "identity" if gui_iterations == 0 else f"x{gui_iterations}"
            if state.debug_mode:
                state.status_msg = (f"Argon2d {profile_label} ({label}) → Stage 2  "
                                    f"Re(p)={state.stage2_p_re:.6f} Im(p)={state.stage2_p_im:.6f}  "
                                    f"Re(o)={state.stage2_o_re:.6f} Im(o)={state.stage2_o_im:.6f}")
            else:
                state.status_msg = f"Argon2d {profile_label} ({label}) → Stage 2"
            state.status_color = CLR_SUCCESS
        except Exception as e:
            state.argon2_digest = ""
            state.status_msg = f"Argon2 error: {e}"
            state.status_color = CLR_ERROR
        state.argon2_running = False

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Stage-2 parameter derivation
# ---------------------------------------------------------------------------

def derive_stage2_params(argon2_digest):
    """Derive second-stage perturbation parameters from the Argon2 digest.

    Returns (o, o_re, o_im, p, p_re, p_im, q, q_re, q_im).
    """
    h = hashlib.sha256(argon2_digest).digest()
    o = struct.unpack('>Q', h[0:ARGON2_INPUT_BYTES])[0]
    p = struct.unpack('>Q', h[ARGON2_INPUT_BYTES:2*ARGON2_INPUT_BYTES])[0]
    q = struct.unpack('>Q', h[2*ARGON2_INPUT_BYTES:3*ARGON2_INPUT_BYTES])[0]
    o_re, o_im = decode_o_display(o)
    p_re, p_im = decode_p_display(p)
    q_re, q_im = decode_q_display(q)
    return o, o_re, o_im, p, p_re, p_im, q, q_re, q_im


def decode_o_display(o):
    """Decode orbit seed o into (Re, Im) floats for display (no baseline)."""
    mag_re = sum((1 if (o & (1 << j)) else 0) * 2.0**(-(O_MAGNITUDE_MIN_EXP + j))
                 for j in range(O_MAGNITUDE_BITS))
    mag_im = sum((1 if (o & (1 << (j + 32))) else 0) * 2.0**(-(O_MAGNITUDE_MIN_EXP + j))
                 for j in range(O_MAGNITUDE_BITS))
    sign_re = -1.0 if (o & (1 << O_SIGN_BIT_RE)) else 1.0
    sign_im = -1.0 if (o & (1 << O_SIGN_BIT_IM)) else 1.0
    return sign_re * mag_re, sign_im * mag_im


def decode_p_display(p):
    """Decode additive perturbation p into (Re, Im) floats for display."""
    baseline = 2.0 ** (-P_BASELINE_EXP)
    mag_re = sum((1 if (p & (1 << j)) else 0) * 2.0**(-(P_MAGNITUDE_MIN_EXP + j))
                 for j in range(P_MAGNITUDE_BITS))
    mag_im = sum((1 if (p & (1 << (j + 32))) else 0) * 2.0**(-(P_MAGNITUDE_MIN_EXP + j))
                 for j in range(P_MAGNITUDE_BITS))
    mag_re += baseline
    mag_im += baseline
    sign_re = -1.0 if (p & (1 << P_SIGN_BIT_RE)) else 1.0
    sign_im = -1.0 if (p & (1 << P_SIGN_BIT_IM)) else 1.0
    return sign_re * mag_re, sign_im * mag_im


def decode_q_display(q):
    """Decode linear perturbation q into (Re, Im) floats for display (no baseline)."""
    mag_re = sum((1 if (q & (1 << j)) else 0) * 2.0**(-(Q_MAGNITUDE_MIN_EXP + j))
                 for j in range(Q_MAGNITUDE_BITS))
    mag_im = sum((1 if (q & (1 << (j + 32))) else 0) * 2.0**(-(Q_MAGNITUDE_MIN_EXP + j))
                 for j in range(Q_MAGNITUDE_BITS))
    sign_re = -1.0 if (q & (1 << Q_SIGN_BIT_RE)) else 1.0
    sign_im = -1.0 if (q & (1 << Q_SIGN_BIT_IM)) else 1.0
    return sign_re * mag_re, sign_im * mag_im


# ---------------------------------------------------------------------------
# F2: random encode pipeline (background thread)
# ---------------------------------------------------------------------------

def run_random_encode(state):
    """F2: generate random entropy bits and run the full encode pipeline.

    Uses state.entropy_bits to determine size (64/128/256).
    Runs in a background thread; updates state progressively.
    """
    profile = state.argon2_profile
    try:
        iters = int(state.argon2_iterations)
        if iters < 0:
            raise ValueError
    except ValueError:
        iters = 0

    state.argon2_running = True
    state.argon2_progress = 0
    state.argon2_progress_total = max(iters, 1)
    prof_label = {PROFILE_BASIC: "Basic", PROFILE_ADVANCED: "Advanced",
                  PROFILE_GREAT_WALL: "Great Wall"}.get(profile, "Basic")
    state.status_msg = f"F2: random encode ({prof_label}, x{iters})..."
    state.status_color = CLR_PENDING

    total_entropy = state.entropy_bits
    bps = state.bits_per_stage

    def _worker():
        try:
            rand_bytes = os.urandom(total_entropy // 8)
            entropy_bits = []
            for b in rand_bytes:
                for j in range(7, -1, -1):
                    entropy_bits.append((b >> j) & 1)
            stage1_bits = entropy_bits[:bps]
            stage2_bits = entropy_bits[bps:]

            # Encode stage 1
            s1_pts, s1_chunks, s1_steps, s1_rects = \
                encode_bits_stage(stage1_bits, STAGE1_O, STAGE1_P, STAGE1_Q)
            state.stage1_encoded_points = s1_pts
            state.stage1_encoded_bits_chunks = s1_chunks
            state.stage1_encoded_steps = s1_steps
            state.stage1_encoded_final_rects = s1_rects
            state.argon2_stage1_bits = stage1_bits
            state.needs_redraw = True

            # Argon2 hash
            data = bits_to_bytes(stage1_bits)
            if iters == 0:
                digest = data.ljust(ARGON2_DIGEST_BYTES, b'\x00')[:ARGON2_DIGEST_BYTES]
                state.argon2_progress = 1
            else:
                input_hex = data.hex()
                ckpt_path = _checkpoint_path(input_hex, profile)
                saved = _load_checkpoint(ckpt_path)
                resume_it = 0
                digest = None
                for it in sorted(saved.keys()):
                    if it <= iters:
                        resume_it = it
                        digest = saved[it]
                if resume_it == 0:
                    digest = argon2_single(data, profile)
                    _save_checkpoint(ckpt_path, 1, digest)
                    state.argon2_progress = 1
                    resume_it = 1
                else:
                    state.argon2_progress = resume_it
                for i in range(resume_it, iters):
                    digest = argon2_single(digest, profile)
                    _save_checkpoint(ckpt_path, i + 1, digest)
                    state.argon2_progress = i + 1

            state.argon2_digest = digest.hex()
            o, o_re, o_im, p, p_re, p_im, q, q_re, q_im = derive_stage2_params(digest)
            state.stage2_o = o
            state.stage2_o_re = o_re
            state.stage2_o_im = o_im
            state.stage2_p = p
            state.stage2_p_re = p_re
            state.stage2_p_im = p_im
            state.stage2_q = q
            state.stage2_q_re = q_re
            state.stage2_q_im = q_im
            state.argon2_marker = argon2_path_marker(profile, iters)
            cache_clear_stage2()

            # Encode stage 2
            s2_pts, s2_chunks, s2_steps, s2_rects = \
                encode_bits_stage(stage2_bits, o, p, q)
            state.stage2_encoded_points = s2_pts
            state.stage2_encoded_bits_chunks = s2_chunks
            state.stage2_encoded_steps = s2_steps
            state.stage2_encoded_final_rects = s2_rects

            # BIP39 mnemonic
            mnemonic = bits_to_mnemonic(entropy_bits)
            state.input_text = mnemonic
            state.input_cursor = len(mnemonic)
            state.input_sel = len(mnemonic)
            state.decoded_mnemonic = mnemonic

            state.stage = 2
            state.needs_redraw = True
            state.selected_point_idx = None
            state.selected_decoded_idx = None
            state.status_msg = f"F2: {mnemonic[:40]}..."
            state.status_color = CLR_SUCCESS
        except Exception as e:
            state.status_msg = f"F2 error: {e}"
            state.status_color = CLR_ERROR
        state.argon2_running = False

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
