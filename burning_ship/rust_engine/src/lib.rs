/// Engine algorithm version.
///
/// Bump MAJOR when the encode/decode output changes for the same input.
/// Bump MINOR for new features that don't change existing outputs.
/// Pre-1.0.0: algorithm is unstable and may change freely.
///
/// 0.2.0: raised the encode/decode island-discovery escape cap (GUI_PARAMS
///        max_iter 64 -> 1024) so deep bisection levels near the set boundary
///        stay navigable instead of starving discovery and stalling.  This is
///        an output-changing (backward-incompatible) algorithm change: 0.1.0
///        encodings do not reproduce across it, so frozen vectors built under
///        0.1.0 are STALE and are rebuilt at the stable release.
/// 0.3.0: fixed a u128 overflow in the island-discovery exclusion-threshold
///        test (discovery.rs `exclusion_threshold_reached`). The excluded/total
///        area ratio is now compared via a deterministic, overflow-safe
///        coarsening (a common magnitude-keyed right-shift on both operands)
///        instead of a raw cross-multiply that overflowed u128 for the full
///        viewport — panicking in debug, silently WRAPPING in release. Fixing
///        the wrapped level-0 stop decision is output-changing: 0.2.0 encodings
///        do not reproduce across it (frozen vectors rebuilt at the stable
///        release). Round-tripping is unaffected — encode and decode apply the
///        identical deterministic test.
pub const ENGINE_VERSION: &str = "0.3.0";

/// Log macro that compiles to nothing unless the `verbose` feature is enabled.
/// Build with `cargo build --release --features verbose` to activate.
#[macro_export]
macro_rules! log_verbose {
    ($($arg:tt)*) => {
        #[cfg(feature = "verbose")]
        eprintln!($($arg)*);
    };
}

pub mod fixed;
pub mod fractal;
pub mod discovery;
pub mod bisect;
pub mod leaf_enum;
pub mod render_cache;
pub mod argon2_hash;
pub mod shamir;
pub mod orbit;
pub mod setup_tiers;
pub mod text;
pub mod protocol;
pub mod ffi;
