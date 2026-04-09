"""
Text input field keyboard handling for the BIP39 input.
"""

import pygame

from constants import CLR_SUCCESS, CLR_ERROR
from encoding import encode_bip39, encode_bip39_stage2
from session import copy_to_clipboard, paste_from_clipboard


# ---------------------------------------------------------------------------
# Word boundary helpers
# ---------------------------------------------------------------------------

def _word_boundary_left(text, pos):
    """Find the start of the word to the left of pos."""
    if pos <= 0:
        return 0
    i = pos - 1
    # skip whitespace
    while i > 0 and text[i] == ' ':
        i -= 1
    # skip word chars
    while i > 0 and text[i - 1] != ' ':
        i -= 1
    return i


def _word_boundary_right(text, pos):
    """Find the end of the word to the right of pos."""
    n = len(text)
    if pos >= n:
        return n
    i = pos
    # skip word chars
    while i < n and text[i] != ' ':
        i += 1
    # skip whitespace
    while i < n and text[i] == ' ':
        i += 1
    return i


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def input_selection(state):
    """Return (lo, hi) of the current selection range."""
    a, b = state.input_cursor, state.input_sel
    return (min(a, b), max(a, b))


def input_delete_selection(state):
    """Delete selected text, collapse cursor. Returns True if there was a selection."""
    lo, hi = input_selection(state)
    if lo == hi:
        return False
    state.input_text = state.input_text[:lo] + state.input_text[hi:]
    state.input_cursor = lo
    state.input_sel = lo
    return True


def input_selected_text(state):
    """Return the currently selected text."""
    lo, hi = input_selection(state)
    return state.input_text[lo:hi]


# ---------------------------------------------------------------------------
# Main keyboard handler
# ---------------------------------------------------------------------------

def handle_text_input(state, event):
    """Handle a KEYDOWN event while the text input is focused."""
    mods = event.mod
    ctrl = mods & (pygame.KMOD_CTRL | pygame.KMOD_META)
    shift = mods & pygame.KMOD_SHIFT
    text = state.input_text
    cur = state.input_cursor

    if event.key == pygame.K_RETURN:
        try:
            if state.stage == 2:
                pts, chunks, steps, frects = encode_bip39_stage2(
                    state.input_text, state.stage2_o, state.stage2_p, state.stage2_q,
                    num_points=state.points_per_stage)
                state.stage2_encoded_points = pts
                state.stage2_encoded_bits_chunks = chunks
                state.stage2_encoded_steps = steps
                state.stage2_encoded_final_rects = frects
                state.selected_point_idx = None
                state.selected_decoded_idx = None
                state.status_msg = f"Encoded {state.points_per_stage} points ({state.bits_per_stage} bits, stage 2)"
                state.status_color = CLR_SUCCESS
                state.needs_redraw = True
            else:
                pts, chunks, steps, frects = encode_bip39(state.input_text,
                    num_points=state.points_per_stage)
                state.stage1_encoded_points = pts
                state.stage1_encoded_bits_chunks = chunks
                state.stage1_encoded_steps = steps
                state.stage1_encoded_final_rects = frects
                state.selected_point_idx = None
                state.selected_decoded_idx = None
                stage1 = []
                for c in chunks:
                    stage1.extend(c)
                state.argon2_stage1_bits = stage1
                state.argon2_digest = ""
                state.status_msg = f"Encoded {state.points_per_stage} points ({state.bits_per_stage} bits, stage 1)"
                state.status_color = CLR_SUCCESS
                state.needs_redraw = True
        except ValueError as e:
            state.status_msg = f"Error: {e}"
            state.status_color = CLR_ERROR
        return

    if event.key == pygame.K_TAB or event.key == pygame.K_ESCAPE:
        state.input_focused = False
        return

    # --- Ctrl shortcuts ---
    if ctrl:
        if event.key == pygame.K_a:
            # Select all
            state.input_sel = 0
            state.input_cursor = len(text)
            return
        if event.key == pygame.K_c:
            sel = input_selected_text(state)
            if sel:
                try:
                    copy_to_clipboard(sel)
                except Exception:
                    pass
            return
        if event.key == pygame.K_x:
            sel = input_selected_text(state)
            if sel:
                try:
                    copy_to_clipboard(sel)
                except Exception:
                    pass
                input_delete_selection(state)
            return
        if event.key == pygame.K_v:
            paste = paste_from_clipboard()
            if paste:
                # Strip newlines — this is a single-line input
                paste = paste.replace("\n", " ").replace("\r", "")
                input_delete_selection(state)
                state.input_text = (state.input_text[:state.input_cursor]
                                    + paste
                                    + state.input_text[state.input_cursor:])
                state.input_cursor += len(paste)
                state.input_sel = state.input_cursor
            return
        # Ctrl+Left: jump word left
        if event.key == pygame.K_LEFT:
            new_pos = _word_boundary_left(text, cur)
            state.input_cursor = new_pos
            if not shift:
                state.input_sel = new_pos
            return
        # Ctrl+Right: jump word right
        if event.key == pygame.K_RIGHT:
            new_pos = _word_boundary_right(text, cur)
            state.input_cursor = new_pos
            if not shift:
                state.input_sel = new_pos
            return
        # Ctrl+Backspace: delete word left
        if event.key == pygame.K_BACKSPACE:
            if not input_delete_selection(state):
                new_pos = _word_boundary_left(text, cur)
                state.input_text = text[:new_pos] + text[cur:]
                state.input_cursor = new_pos
                state.input_sel = new_pos
            return
        # Ctrl+Delete: delete word right
        if event.key == pygame.K_DELETE:
            if not input_delete_selection(state):
                new_pos = _word_boundary_right(text, cur)
                state.input_text = text[:cur] + text[new_pos:]
                state.input_sel = state.input_cursor
            return
        return  # ignore other Ctrl combos

    # --- Navigation keys ---
    if event.key == pygame.K_LEFT:
        has_sel = cur != state.input_sel
        if not shift and has_sel:
            state.input_cursor = min(cur, state.input_sel)
        elif cur > 0:
            state.input_cursor = cur - 1
        if not shift:
            state.input_sel = state.input_cursor
        return

    if event.key == pygame.K_RIGHT:
        has_sel = cur != state.input_sel
        if not shift and has_sel:
            state.input_cursor = max(cur, state.input_sel)
        elif cur < len(text):
            state.input_cursor = cur + 1
        if not shift:
            state.input_sel = state.input_cursor
        return

    if event.key == pygame.K_HOME:
        state.input_cursor = 0
        if not shift:
            state.input_sel = 0
        return

    if event.key == pygame.K_END:
        state.input_cursor = len(text)
        if not shift:
            state.input_sel = len(text)
        return

    # --- Deletion ---
    if event.key == pygame.K_BACKSPACE:
        if not input_delete_selection(state):
            if cur > 0:
                state.input_text = text[:cur - 1] + text[cur:]
                state.input_cursor = cur - 1
                state.input_sel = state.input_cursor
        return

    if event.key == pygame.K_DELETE:
        if not input_delete_selection(state):
            if cur < len(text):
                state.input_text = text[:cur] + text[cur + 1:]
                state.input_sel = state.input_cursor
        return

    # --- Printable character ---
    if event.unicode and event.unicode.isprintable():
        input_delete_selection(state)
        state.input_text = (state.input_text[:state.input_cursor]
                            + event.unicode
                            + state.input_text[state.input_cursor:])
        state.input_cursor += len(event.unicode)
        state.input_sel = state.input_cursor
