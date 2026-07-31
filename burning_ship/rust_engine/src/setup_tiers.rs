//! Canonical setup-tier table for the `0.4.0` orbit protocol.
//!
//! The **setup pathway** (`great-wall-docs/great-wall-core/DESIGN.md` *Setup
//! Pathway*; TGPO §6–§7) is a sequence of progressive builds where every prefix
//! is itself a valid setup. Each setup **level** fixes, per stage, the Shamir
//! threshold `t_i` — which equals the entropy-bearing fractal count `r_i` — and
//! the stage carries `t_i * 32` bits. This module is the engine-authoritative
//! source of that table so callers read tiers from the engine rather than
//! hand-mirroring them.
//!
//! Rules:
//! - **Stage 0** (the `σ`-public entry, assumed seizable per TM.9) is always
//!   `t_0 = 2` ([`STAGE0_THRESHOLD`]).
//! - **Deep (non-zero) stages** are STANDARD at `t_i = 3` (≥ 96 bits): the rule
//!   is `r_i > 2` ([`STANDARD_THRESHOLD`]).
//! - The **sole exception** is the entry-level **Setup 1**, whose single deep
//!   stage runs `t_1 = 2` (64 bits) — the model's softest number, flagged
//!   **substandard** ([`ENTRY_THRESHOLD`]).
//!
//! Per-level thresholds `t_i` (index 0 = stage 0, 1..=N = deep stages):
//! ```text
//!   level 1  -> [2 | 2]         1 deep stage   (SUBSTANDARD, 64-bit)
//!   level 2  -> [2 | 3]         1 deep stage   (standard, 96-bit)
//!   level 3  -> [2 | 3, 3]      2 deep stages
//!   level k  -> [2 | 3 × (k-1)] (k-1) deep stages
//! ```
//! The Setup 1 → 2 step is the only one that *upgrades a stage in place*
//! (`t_1: 2 → 3`); every later step *appends* a standard deep stage.

/// Stage-0 threshold: the two `σ`-public entry points (TM.9).
pub const STAGE0_THRESHOLD: u32 = 2;
/// Standard deep-stage threshold — the `r_i > 2` rule: 3 fractals, ≥ 96 bits.
pub const STANDARD_THRESHOLD: u32 = 3;
/// Entry-level (Setup 1) deep-stage threshold: 2 fractals, 64 bits — substandard.
pub const ENTRY_THRESHOLD: u32 = 2;

/// Upper bound on the setup level the table will materialise (a DoS guard for
/// the FFI: `thresholds` allocates `O(level)`). A setup with dozens of deep
/// stages is already far past any real use.
pub const MAX_SETUP_LEVEL: u32 = 64;

/// Per-stage thresholds `t_i` for a setup `level` (1-based): index 0 is stage 0,
/// indices `1..=N` are the deep stages. Length is `N + 1`.
///
/// Panics if `level == 0` (levels are 1-based) — Rust callers pass valid levels;
/// the FFI guards `level` before calling so this never unwinds across the ABI.
pub fn thresholds(level: u32) -> Vec<u32> {
    assert!(level >= 1, "setup level is 1-based (>= 1)");
    let mut t = Vec::with_capacity(level as usize);
    t.push(STAGE0_THRESHOLD);
    if level == 1 {
        t.push(ENTRY_THRESHOLD); // one substandard deep stage
    } else {
        for _ in 0..(level - 1) {
            t.push(STANDARD_THRESHOLD); // (level - 1) standard deep stages
        }
    }
    t
}

/// Number of deep (non-zero) stages `N` at `level`.
pub fn deep_stages(level: u32) -> u32 {
    (thresholds(level).len() - 1) as u32
}

/// Whether `level` is substandard (only the entry tier, Setup 1).
pub fn is_substandard(level: u32) -> bool {
    level == 1
}

/// Total tacit entropy in bits across the deep stages at `level`
/// (`Σ_{i≥1} t_i · 32`).
pub fn entropy_bits(level: u32) -> u32 {
    thresholds(level)[1..].iter().sum::<u32>() * 32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_tiers() {
        assert_eq!(thresholds(1), vec![2, 2]);
        assert_eq!(thresholds(2), vec![2, 3]);
        assert_eq!(thresholds(3), vec![2, 3, 3]);
        assert_eq!(thresholds(4), vec![2, 3, 3, 3]);
    }

    #[test]
    fn deep_stage_counts_and_entropy() {
        assert_eq!((deep_stages(1), entropy_bits(1)), (1, 64)); // entry
        assert_eq!((deep_stages(2), entropy_bits(2)), (1, 96)); // standard
        assert_eq!((deep_stages(3), entropy_bits(3)), (2, 192));
        assert_eq!((deep_stages(4), entropy_bits(4)), (3, 288));
    }

    #[test]
    fn only_setup_1_is_substandard() {
        assert!(is_substandard(1));
        for level in 2..=MAX_SETUP_LEVEL {
            assert!(!is_substandard(level), "level {level} must be standard");
        }
    }

    #[test]
    fn the_r_i_gt_2_rule_holds_except_entry() {
        // Every deep threshold is > 2 for standard setups; exactly the entry
        // tier has a deep threshold == 2, and it is the one flagged substandard.
        for level in 1..=MAX_SETUP_LEVEL {
            let t = thresholds(level);
            let deep_ok = t[1..].iter().all(|&ti| ti > 2);
            assert_eq!(deep_ok, !is_substandard(level),
                "level {level}: deep-threshold>2 must match not-substandard");
            assert_eq!(t[0], 2, "stage 0 threshold is always 2");
        }
    }

    #[test]
    fn pathway_appends_a_standard_stage_after_setup_2() {
        // Setup 1 -> 2 upgrades the deep stage in place (2 -> 3); from Setup 2
        // on, each level appends one standard deep stage.
        assert_eq!(thresholds(2), vec![2, 3]);
        for k in 3..=MAX_SETUP_LEVEL {
            let mut expected = thresholds(k - 1);
            expected.push(STANDARD_THRESHOLD);
            assert_eq!(thresholds(k), expected, "level {k} appends one t=3 stage");
        }
    }
}
