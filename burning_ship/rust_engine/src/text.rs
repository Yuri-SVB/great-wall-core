//! Stage-text canonicalization and chain-input assembly — the single source of
//! truth shared with great-wallet over FFI (`bs_salt_pepper_canonicalize` /
//! `bs_chain_input`).
//!
//! Faithful port of `normalize_stage_text` / `stage_text_bytes` /
//! `bits_to_bytes` in great-wall-core/burning_ship/encoding.py (DESIGN.md
//! "Strong text restrictions"). Every stage text input — the stage-0
//! salt/pepper and every non-0 stage's export label — is restricted to
//! upper-case ASCII alphanumerics and `-` only (`[A-Z0-9-]`) so the same text
//! round-trips identically across devices, keyboards, locales, and clipboards.
//! A stray lower-case letter, accent, or Unicode look-alike would otherwise
//! silently fork the chain into a different, unrecoverable result.

/// True if `b` is in the stage-text alphabet `[A-Z0-9-]`.
fn in_alphabet(b: u8) -> bool {
    b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'-'
}

/// Canonicalize raw stage text to the `[A-Z0-9-]` set: up-case ASCII letters
/// and drop every byte outside the alphabet (accents, spaces, punctuation,
/// control bytes, and — because the input is treated as bytes — any non-ASCII).
/// Returns the canonical ASCII bytes.
pub fn canonicalize_stage_text(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for &b in text {
        let up = b.to_ascii_uppercase();
        if in_alphabet(up) {
            out.push(up);
        }
    }
    out
}

/// Pack 0/1 bytes MSB-first into bytes; the final byte is zero-padded on the
/// right. Mirrors `bits_to_bytes` in encoding.py.
pub fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let n = bits.len();
    let mut out = Vec::with_capacity(n.div_ceil(8));
    let mut i = 0;
    while i < n {
        let mut byte: u8 = 0;
        for j in 0..8 {
            byte <<= 1;
            if i + j < n && bits[i + j] != 0 {
                byte |= 1;
            }
        }
        out.push(byte);
        i += 8;
    }
    out
}

/// Build one chain link's Argon2 input: the canonical stage-0 text bytes
/// followed by the packed prior-point bits. Mirrors
/// `stage_text_bytes(text) + bits_to_bytes(prior_bits)` in
/// `argon2_pipeline.derive_stage_params` — the protocol byte layout that seeds
/// every point stage's fractal.
pub fn chain_input(text: &[u8], prior_bits: &[u8]) -> Vec<u8> {
    let mut out = canonicalize_stage_text(text);
    out.extend_from_slice(&bits_to_bytes(prior_bits));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upcases_and_drops() {
        assert_eq!(canonicalize_stage_text(b"main-stash"), b"MAIN-STASH");
        assert_eq!(canonicalize_stage_text(b"My Stash!"), b"MYSTASH");
        assert_eq!(canonicalize_stage_text(b"a_b.c"), b"ABC");
        assert_eq!(canonicalize_stage_text(b""), b"");
        // Non-ASCII bytes (e.g. UTF-8 for an accent) are dropped.
        assert_eq!(canonicalize_stage_text("café".as_bytes()), b"CAF");
    }

    #[test]
    fn keeps_canonical_text_unchanged() {
        let t = b"RETIREMENT-2040";
        assert_eq!(canonicalize_stage_text(t), t);
    }

    #[test]
    fn bits_pack_msb_first_with_padding() {
        assert_eq!(bits_to_bytes(&[]), b"");
        assert_eq!(bits_to_bytes(&[1, 0, 0, 0, 0, 0, 0, 1]), &[0x81]);
        // 0b101 -> left-aligned in one byte: 0b10100000 = 0xA0
        assert_eq!(bits_to_bytes(&[1, 0, 1]), &[0xA0]);
        // 9 bits -> 2 bytes, second byte zero-padded on the right.
        assert_eq!(bits_to_bytes(&[1; 9]), &[0xFF, 0x80]);
    }

    #[test]
    fn chain_input_is_text_then_packed_bits() {
        let out = chain_input(b"ab", &[1, 0, 1, 0, 0, 0, 0, 0]);
        assert_eq!(out, &[b'A', b'B', 0xA0]);
        // No prior bits -> just the canonical text.
        assert_eq!(chain_input(b"ab", &[]), b"AB");
    }
}
