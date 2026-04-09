"""
Manual bit-input mode: incremental encoding with O/I keys.
"""

from burning_ship_engine import encode, encode_with_seeds
from bip39 import bits_to_mnemonic
from constants import (
    BITS_PER_POINT, ENCODE_AREA, GUI_PARAMS,
    STAGE1_O, STAGE1_P, STAGE1_Q,
    CLR_BIT_OK, CLR_ADVANCE, CLR_SUCCESS,
)
from encoding import bits_to_hex


def manual_update_viz(state):
    """Update area visualization from current manual_encode_result + committed data."""
    r = state.manual_encode_result
    if r:
        all_pts = list(state.manual_committed_points) + [
            (r.point_re, r.point_im, r.point_re_raw, r.point_im_raw)]
        # Use merged steps (incremental) if available, else full steps.
        steps = r._merged_steps if hasattr(r, '_merged_steps') and r._merged_steps is not None else r.get_all_steps()
        all_steps = list(state.manual_committed_steps) + [steps]
        all_rects = list(state.manual_committed_rects) + [r.final_rect]
    else:
        all_pts = list(state.manual_committed_points)
        all_steps = list(state.manual_committed_steps)
        all_rects = list(state.manual_committed_rects)
    if state.stage == 2:
        state.stage2_encoded_points = all_pts
        state.stage2_encoded_steps = all_steps
        state.stage2_encoded_final_rects = all_rects
    else:
        state.stage1_encoded_points = all_pts
        state.stage1_encoded_steps = all_steps
        state.stage1_encoded_final_rects = all_rects


def manual_encode_latest(state):
    """Incrementally encode the latest manual bit.

    Uses inherited_seeds from the previous encode result to ensure the
    area tree matches a full from-scratch encode or decode_full.
    Falls back to full encode for the first bit or after undo to empty.
    """
    if not state.manual_bits:
        state.manual_encode_result = None
        manual_update_viz(state)
        return
    o_val = state.stage2_o if (state.stage == 2 and state.stage2_o is not None) else STAGE1_O
    p_val = state.stage2_p if (state.stage == 2 and state.stage2_p is not None) else STAGE1_P
    q_val = state.stage2_q if (state.stage == 2 and state.stage2_q is not None) else STAGE1_Q

    prev = state.manual_encode_history[-1] if state.manual_encode_history else None
    if prev is not None and len(state.manual_bits) > 1:
        # Incremental: encode only the last bit, starting from the
        # previous result's leaf rect, path, and inherited seeds.
        new_bit = [state.manual_bits[-1]]
        seeds_blob, num_seeds = prev.get_seeds_blob()
        incr = encode_with_seeds(new_bit, seeds_blob, num_seeds,
                                 area=prev.final_rect, params=GUI_PARAMS,
                                 o=o_val, p=p_val, q=q_val,
                                 path_prefix=prev.path)
        # Build a merged result: all prior steps + the new step.
        prev_steps = prev._merged_steps if hasattr(prev, '_merged_steps') and prev._merged_steps is not None else prev.get_all_steps()
        merged_steps = prev_steps + incr.get_all_steps()
        incr._merged_steps = merged_steps
        state.manual_encode_result = incr
    else:
        # First bit or reset — full encode from scratch.
        result = encode(state.manual_bits, area=ENCODE_AREA, params=GUI_PARAMS,
                        o=o_val, p=p_val, q=q_val, path_prefix="O")
        result._merged_steps = None
        state.manual_encode_result = result
    manual_update_viz(state)


def manual_commit_point(state):
    """Commit the current 32-bit point and advance to the next."""
    r = state.manual_encode_result
    state.manual_committed_points.append(
        (r.point_re, r.point_im, r.point_re_raw, r.point_im_raw))
    steps = r._merged_steps if hasattr(r, '_merged_steps') and r._merged_steps is not None else r.get_all_steps()
    state.manual_committed_steps.append(steps)
    state.manual_committed_rects.append(r.final_rect)
    state.manual_committed_bits.append(list(state.manual_bits))
    state.manual_point_idx += 1
    state.manual_bits = []
    state.manual_encode_result = None
    state.manual_encode_history = []


def manual_add_bit(state, bit):
    """Add a bit in manual input mode and re-encode."""
    if len(state.manual_bits) >= BITS_PER_POINT:
        return  # should not happen — auto-commit handles this
    # Save current state for instant undo
    state.manual_encode_history.append(state.manual_encode_result)
    state.manual_bits.append(bit)
    manual_encode_latest(state)
    r = state.manual_encode_result
    n = len(state.manual_bits)
    pt = state.manual_point_idx + 1
    path = r.path if r else "O"
    state.status_msg = f"P{pt} bit {n}/{BITS_PER_POINT}: {'I' if bit else 'O'} → path={path}"
    state.status_color = CLR_BIT_OK
    state.show_areas = True
    state.area_focus_step = None  # show entire bisection path

    # Auto-commit at BITS_PER_POINT
    if n == BITS_PER_POINT:
        manual_commit_point(state)
        if state.manual_point_idx < state.points_per_stage:
            pt_done = state.manual_point_idx
            state.status_msg = f"Point {pt_done} done ({BITS_PER_POINT} bits). Now enter point {pt_done + 1}."
            state.status_color = CLR_ADVANCE
        else:
            manual_finish_stage(state)


def manual_finish_stage(state):
    """Finish manual mode after all points are committed."""
    pts = state.manual_committed_points
    steps = state.manual_committed_steps
    rects = state.manual_committed_rects
    all_bits = []
    for chunk in state.manual_committed_bits:
        all_bits.extend(chunk)

    if state.stage == 2:
        state.stage2_encoded_points = list(pts)
        state.stage2_encoded_steps = list(steps)
        state.stage2_encoded_final_rects = list(rects)
        if state.argon2_stage1_bits is not None:
            full_entropy = state.argon2_stage1_bits + all_bits
            mnemonic = bits_to_mnemonic(full_entropy)
            state.decoded_mnemonic = mnemonic
        else:
            state.decoded_mnemonic = f"[stage 2: {bits_to_hex(all_bits)}] (stage 1 missing)"
    else:
        state.stage1_encoded_points = list(pts)
        state.stage1_encoded_steps = list(steps)
        state.stage1_encoded_final_rects = list(rects)
        state.argon2_stage1_bits = all_bits
        state.decoded_stage1_bits = all_bits
        state.argon2_digest = ""
    state.manual_bits_mode = False
    state.status_msg = f"Manual encode done: {state.points_per_stage}\u00d7{BITS_PER_POINT} bits"
    state.status_color = CLR_SUCCESS
