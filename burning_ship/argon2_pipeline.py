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
    P_MAGNITUDE_BITS, P_SIGN_BIT_RE, P_SIGN_BIT_IM,
    P_MAGNITUDE_MIN_EXP, P_BASELINE_EXP,
    Q_MAGNITUDE_BITS, Q_SIGN_BIT_RE, Q_SIGN_BIT_IM, Q_MAGNITUDE_MIN_EXP,
    O_MAGNITUDE_BITS, O_SIGN_BIT_RE, O_SIGN_BIT_IM, O_MAGNITUDE_MIN_EXP,
    CLR_PENDING, CLR_SUCCESS, CLR_ERROR, CLR_WARNING,
)
from encoding import (
    argon2_path_marker, bits_to_bytes, stage_text_bytes,
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

class _Argon2Stopped(Exception):
    """Raised in the worker thread when state.argon2_stop_requested goes True."""


def _check_argon2_stop(state):
    """Raise _Argon2Stopped if the main thread requested cancellation.

    Stop granularity is one Argon2d iteration: the in-flight Rust call is not
    interruptible, so this is checked between iterations.
    """
    if getattr(state, "argon2_stop_requested", False):
        state.argon2_stop_requested = False
        raise _Argon2Stopped()


def argon2_iterate(data, profile, iterations, progress_cb=None, stop_check=None):
    """Run ``iterations`` sequential Argon2d passes over ``data``.

    Each pass feeds the previous 32-byte digest back as the next input — a
    single sequential, memory-hard chain (the wall-clock cost the protocol is
    built on).  ``iterations == 0`` is the identity transform: the input is
    zero-padded/truncated to the digest width and returned unchanged.

    ``progress_cb(done)`` (optional) is invoked after every completed pass, and
    ``stop_check()`` (optional) is invoked after every pass and may raise to
    abort the chain.  Returns the final 32-byte digest.

    This is the one place the Argon2 iteration loop lives; the GUI threads and
    the deterministic CLI/protocol paths all call through here.
    """
    if iterations == 0:
        return data.ljust(ARGON2_DIGEST_BYTES, b'\x00')[:ARGON2_DIGEST_BYTES]
    digest = argon2_single(data, profile)
    if progress_cb is not None:
        progress_cb(1)
    if stop_check is not None:
        stop_check()
    for i in range(1, iterations):
        digest = argon2_single(digest, profile)
        if progress_cb is not None:
            progress_cb(i + 1)
        if stop_check is not None:
            stop_check()
    return digest


def derive_stage_params(stage0_text, prior_bits, profile, iterations,
                        progress_cb=None, stop_check=None):
    """Derive a point stage's fractal parameters from stage-0 text + prior points.

    Under the protocol 0.3.0 chain (DESIGN.md "Chained Protocol"), every point
    stage's fractal (o, p, q) is the memory-hard hash of **stage-0 text followed
    by the concatenated bits of every point that precedes it**::

        θ = SHA-256( Argon2^iterations( stage-0 text ‖ prior point bits ) ) → (o, p, q)

    The first point stage (no prior points) therefore derives from stage-0 text
    alone — so no fractal in the chain is the old public "canonical" surface.
    Because the input grows by one point per stage and each link is a full Argon2
    chain, the cost of descriptively bypassing the derivation compounds.

    Returns ``(digest, params)`` where ``digest`` is the 32-byte Argon2 output
    and ``params`` is the 9-tuple from :func:`derive_stage2_params`.
    """
    data = stage_text_bytes(stage0_text) + bits_to_bytes(prior_bits)
    digest = argon2_iterate(data, profile, iterations, progress_cb, stop_check)
    return digest, derive_stage2_params(digest)


def _next_underived_stage(state):
    """Return the lowest point-stage index whose fractal is not yet derived.

    Under protocol 0.3.0 every point stage (including the first, index 0) is
    chain-derived from stage-0 text + preceding points, so the GUI derives one
    stage at a time starting at index 0.  Returns ``None`` if every stage
    already has params (nothing left to derive).
    """
    for k in range(0, state.n_stages):
        if state.stages_params[k] is None:
            return k
    return None


def run_argon2_iterative(state, gui_iterations):
    """Derive the NEXT stage's fractal in a background thread.

    Under the one-point-per-stage protocol, this reads the cumulative
    prior-point bits (``state.argon2_stage1_bits``, set by the GUI after each
    point is fixed), runs ``gui_iterations`` Argon2d passes over them, derives
    the next un-derived stage's (o, p, q), stores them into
    ``state.stages_params[next_stage]``, and advances ``state.stage``.
    ``gui_iterations=0`` means identity (no Argon2).

    When state.argon2_save_intermediate is True, intermediate digests are
    checkpointed to disk so the computation can be resumed after interruption
    or so the user can probe nearby iteration counts cheaply.  When False
    (the default), no on-disk state is written or read — preserving the full
    time-cost barrier that Argon2 derivation is meant to impose.

    SECURITY: when this flag is on, any leftover checkpoint file lets an
    attacker skip the wall-clock cost.  It is the user's responsibility to
    delete it (securely) once it has served its purpose.
    """
    prior_bits = state.argon2_stage1_bits
    profile = state.argon2_profile
    # Stage-0 text seeds every derivation (protocol 0.3.0).
    stage0_prefix = stage_text_bytes(getattr(state, "stage0_text", ""))
    save_intermediate = getattr(state, "argon2_save_intermediate", False)
    next_stage = _next_underived_stage(state)
    state.argon2_stop_requested = False

    def _worker():
        try:
            if next_stage is None:
                state.status_msg = "All stages already derived"
                state.status_color = CLR_WARNING
                state.argon2_running = False
                return
            data = stage0_prefix + bits_to_bytes(prior_bits or [])
            if gui_iterations == 0:
                digest = data.ljust(ARGON2_DIGEST_BYTES, b'\x00')[:ARGON2_DIGEST_BYTES]
                state.argon2_progress = 1
            elif save_intermediate:
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
                    _check_argon2_stop(state)
                else:
                    state.argon2_progress = resume_it

                for i in range(resume_it, gui_iterations):
                    digest = argon2_single(digest, profile)
                    cur_it = i + 1
                    _save_checkpoint(ckpt_path, cur_it, digest)
                    state.argon2_progress = cur_it
                    _check_argon2_stop(state)
            else:
                digest = argon2_single(data, profile)
                state.argon2_progress = 1
                _check_argon2_stop(state)
                for i in range(1, gui_iterations):
                    digest = argon2_single(digest, profile)
                    state.argon2_progress = i + 1
                    _check_argon2_stop(state)

            state.argon2_digest = digest.hex()
            params = derive_stage2_params(digest)
            o, o_re, o_im, p, p_re, p_im, q, q_re, q_im = params
            state.stages_params[next_stage] = {
                "o": o, "o_re": o_re, "o_im": o_im,
                "p": p, "p_re": p_re, "p_im": p_im,
                "q": q, "q_re": q_re, "q_im": q_im,
            }
            state.argon2_marker = argon2_path_marker(profile, gui_iterations)
            cache_clear_stage2()
            state.stage = next_stage
            state.needs_redraw = True
            _PROFILE_LABELS = {PROFILE_BASIC: "Basic", PROFILE_ADVANCED: "Advanced",
                               PROFILE_GREAT_WALL: "Great Wall"}
            profile_label = _PROFILE_LABELS.get(profile, "Basic")
            label = "identity" if gui_iterations == 0 else f"x{gui_iterations}"
            stage_lbl = f"Stage {next_stage + 1}/{state.n_stages}"
            if state.debug_mode:
                state.status_msg = (f"Argon2d {profile_label} ({label}) → {stage_lbl}  "
                                    f"Re(o)={o_re:.6f} Im(o)={o_im:.6f}  "
                                    f"Re(p)={p_re:.6f} Im(p)={p_im:.6f}  "
                                    f"Re(q)={q_re:.6f} Im(q)={q_im:.6f}")
            else:
                state.status_msg = f"Argon2d {profile_label} ({label}) → {stage_lbl}"
            state.status_color = CLR_SUCCESS
        except _Argon2Stopped:
            state.argon2_digest = ""
            state.status_msg = (f"Argon2 stopped at iteration "
                                f"{state.argon2_progress}/{gui_iterations}")
            state.status_color = CLR_WARNING
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
    """Decode orbit-seed entropy reservoir o into (Re(o), Im(o)) floats — no baseline."""
    mag_re = sum((1 if (o & (1 << j)) else 0) * 2.0**(-(O_MAGNITUDE_MIN_EXP + j))
                 for j in range(O_MAGNITUDE_BITS))
    mag_im = sum((1 if (o & (1 << (j + 32))) else 0) * 2.0**(-(O_MAGNITUDE_MIN_EXP + j))
                 for j in range(O_MAGNITUDE_BITS))
    sign_re = -1.0 if (o & (1 << O_SIGN_BIT_RE)) else 1.0
    sign_im = -1.0 if (o & (1 << O_SIGN_BIT_IM)) else 1.0
    return sign_re * mag_re, sign_im * mag_im


def decode_p_display(p):
    """Decode additive-perturbation entropy reservoir p into (Re(p), Im(p)) floats —
    baseline 2^{-P_BASELINE_EXP} steers p away from the canonical-formula tail."""
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
    """Decode linear-perturbation entropy reservoir q into (Re(q), Im(q)) floats — no baseline."""
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
    state.argon2_stop_requested = False
    state.argon2_progress = 0
    state.argon2_progress_total = max(iters, 1)
    prof_label = {PROFILE_BASIC: "Basic", PROFILE_ADVANCED: "Advanced",
                  PROFILE_GREAT_WALL: "Great Wall"}.get(profile, "Basic")
    state.status_msg = f"F2: random encode ({prof_label}, x{iters})..."
    state.status_color = CLR_PENDING

    total_entropy = state.entropy_bits
    # Progress spans every point stage (N derivations — every stage is derived
    # in 0.3.0), each of `iters` Argon2 passes; map per-stage per-iteration
    # progress onto the bar.
    n_derivations = max(1, state.n_stages)
    state.argon2_progress_total = max(iters, 1) * n_derivations

    def _worker():
        try:
            import protocol  # GUI orchestration import; deferred to avoid cycles.
            rand_bytes = os.urandom(total_entropy // 8)
            entropy_bits = []
            for b in rand_bytes:
                for j in range(7, -1, -1):
                    entropy_bits.append((b >> j) & 1)

            per_stage = max(iters, 1)

            def _progress(stage_index, done):
                # stage_index is the 0-based point stage (0..N-1).
                base = stage_index * per_stage
                state.argon2_progress = base + done

            def _stop():
                _check_argon2_stop(state)

            # Drive the full chained encode (N memory-hard derivations, one per
            # point stage); stage-0 text seeds the chain.
            stages = protocol.encode_entropy(
                entropy_bits, getattr(state, "stage0_text", ""), profile, iters,
                progress_cb=_progress, stop_check=_stop)

            for sr in stages:
                i = sr.index
                state.stages_encoded_points[i] = [sr.point]
                state.stages_encoded_bits_chunks[i] = [sr.chunk]
                state.stages_encoded_steps[i] = [sr.result.get_all_steps()]
                state.stages_encoded_final_rects[i] = [sr.result.final_rect]
                if sr.params is None:
                    state.stages_params[i] = None
                else:
                    o, o_re, o_im, p, p_re, p_im, q, q_re, q_im = sr.params
                    state.stages_params[i] = {
                        "o": o, "o_re": o_re, "o_im": o_im,
                        "p": p, "p_re": p_re, "p_im": p_im,
                        "q": q, "q_re": q_re, "q_im": q_im,
                    }
                if sr.digest is not None:
                    state.argon2_digest = sr.digest.hex()

            state.argon2_stage1_bits = list(entropy_bits)
            state.argon2_marker = argon2_path_marker(profile, iters)
            cache_clear_stage2()

            # BIP39 mnemonic
            mnemonic = bits_to_mnemonic(entropy_bits)
            state.input_text = mnemonic
            state.input_cursor = len(mnemonic)
            state.input_sel = len(mnemonic)
            state.decoded_mnemonic = mnemonic

            state.stage = 0
            state.needs_redraw = True
            state.selected_point_idx = None
            state.selected_decoded_idx = None
            state.status_msg = f"F2: {mnemonic[:40]}..."
            state.status_color = CLR_SUCCESS
        except _Argon2Stopped:
            state.status_msg = (f"F2 stopped at iteration "
                                f"{state.argon2_progress}/{iters}")
            state.status_color = CLR_WARNING
        except Exception as e:
            state.status_msg = f"F2 error: {e}"
            state.status_color = CLR_ERROR
        state.argon2_running = False

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
