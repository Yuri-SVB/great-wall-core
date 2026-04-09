/// Argon2d hashing for the Great Wall protocol.
///
/// Three fixed profiles control the Argon2 parameters:
///
/// | Profile        | `m` (memory) | `p` (parallelism) | `t` (time cost) | Target device          |
/// |----------------|-------------:|--------------------|------------------|------------------------|
/// | **Basic**      |        1 GiB | 1                  | 32               | Mobile (3-4 GB+)       |
/// | **Advanced**   |       32 GiB | 1                  | 4                | Desktop (48-64 GB)     |
/// | **Great Wall** |      128 GiB | 1                  | 2                | Server (192+ GB)       |
///
/// All produce 32-byte (256-bit) digests using Argon2d v0x13 with salt
/// `b"greatwall"`.  The user controls only the number of *external*
/// iterations (hash-then-feed-back cycles); all other parameters are fixed
/// per-profile.
///
/// ## Why p=1 (single lane) for all tiers
///
/// Argon2 supports p>1 for password-hashing scenarios where the defender has
/// a hard wall-clock budget (e.g. ~1s per login).  More lanes let the defender
/// pack more total work into that fixed time window.
///
/// Great Wall inverts this: the delay IS the product.  If the defender has 4
/// cores and p=4, they finish 4× faster — then we just 4× the outer iteration
/// count to fill the same target delay.  Net result: same wall-clock time,
/// same total work, but the attacker's GPU/ASIC now has 4 independent lanes
/// to exploit in parallel.
///
/// With p=1, the entire computation is a single sequential chain of
/// memory-dependent operations.  This maximises two security metrics:
///
/// 1. **OOM gate**: total memory = p × m.  With p=1 the OOM threshold equals
///    m itself — the full memory cost cannot be reduced by serialising lanes.
/// 2. **ASIC resistance**: an attacker cannot parallelise *within* a single
///    lane (each step depends on the previous one).  The defender's commodity
///    CPU is near-optimal for sequential memory-hard work, minimising the
///    attacker/defender performance ratio.
///
/// ## Time cost (t) per profile
///
/// With t=1, Argon2 fills the memory array in a single sequential pass.
/// Once block i+1 is computed, block i is never read again — an attacker
/// can exploit this by discarding blocks and recomputing them on demand,
/// trading time for memory (the classic time-memory tradeoff).
///
/// With t≥2, subsequent passes re-read the *entire* array in a
/// data-dependent pattern (Argon2d).  Every block from pass 1 might be
/// referenced at any point during pass 2, so the attacker must keep the
/// full m allocation resident at the pass boundary or face a quadratic
/// (or worse) recomputation penalty.
///
/// The time cost is inversely scaled with memory to keep wall-clock time
/// roughly comparable across profiles:
///
/// - **Basic (1 GiB, t=32)**: high t compensates for the smaller memory,
///   ensuring the OOM gate still imposes meaningful sequential work.
/// - **Advanced (32 GiB, t=4)**: moderate t; the large memory footprint
///   already provides strong memory-hardness.
/// - **Great Wall (128 GiB, t=2)**: minimum t needed to guarantee full
///   memory residency; the enormous allocation dominates cost.
///
/// ## Why Argon2d (not Argon2i or Argon2id)
///
/// Argon2i uses data-independent memory access patterns to resist
/// side-channel attacks — critical when the secret is a short password
/// typed into a shared server.  Argon2id is a hybrid that uses Argon2i for
/// the first pass and Argon2d thereafter.
///
/// In Great Wall the "secret" is 64 bits of entropy derived from a BIP-39
/// mnemonic, processed on the user's own device.  Side-channel resistance
/// is irrelevant: the user controls the hardware, and 64-bit entropy
/// cannot be brute-forced even with full knowledge of access patterns.
///
/// Argon2d's data-dependent addressing makes its memory-hardness strictly
/// stronger: the access pattern depends on the data, so an attacker cannot
/// precompute which memory cells will be needed and must keep the full
/// allocation resident.  This maximises the memory-hardness guarantee —
/// exactly what we want for the OOM gate.

use argon2::{Algorithm, Argon2, Params, Version};

/// Profile identifier.  0 = Basic, 1 = Advanced, 2 = Great Wall.
pub type Profile = u8;
pub const PROFILE_BASIC: Profile = 0;
pub const PROFILE_ADVANCED: Profile = 1;
pub const PROFILE_GREAT_WALL: Profile = 2;

// --- Basic profile (mobile-accessible) ---
const BASIC_MEMORY_KIB: u32 = 1_048_576;    // 1 GiB
const BASIC_TIME_COST: u32 = 32;
const BASIC_PARALLELISM: u32 = 1;

// --- Advanced profile (desktop-only) ---
const ADV_MEMORY_KIB: u32 = 33_554_432;     // 32 GiB
const ADV_TIME_COST: u32 = 4;
const ADV_PARALLELISM: u32 = 1;

// --- Great Wall profile (server-class, wrench-attack resistant) ---
const GW_MEMORY_KIB: u32 = 134_217_728;     // 128 GiB
const GW_TIME_COST: u32 = 2;
const GW_PARALLELISM: u32 = 1;

// --- Common ---
const ARGON2_HASH_LEN: usize = 32;      // 256-bit digest
const ARGON2_SALT: &[u8] = b"greatwall";

fn make_argon2(profile: Profile) -> Argon2<'static> {
    let (mem, time, par) = match profile {
        PROFILE_ADVANCED   => (ADV_MEMORY_KIB,   ADV_TIME_COST,   ADV_PARALLELISM),
        PROFILE_GREAT_WALL => (GW_MEMORY_KIB,    GW_TIME_COST,    GW_PARALLELISM),
        _                  => (BASIC_MEMORY_KIB,  BASIC_TIME_COST, BASIC_PARALLELISM),
    };
    let params = Params::new(mem, time, par, Some(ARGON2_HASH_LEN))
        .expect("valid Argon2 params");
    Argon2::new(Algorithm::Argon2d, Version::V0x13, params)
}

/// Run a single Argon2d pass on arbitrary-length input.
/// Returns 32 bytes (256-bit digest).
pub fn argon2_single(input: &[u8], profile: Profile) -> [u8; ARGON2_HASH_LEN] {
    let argon2 = make_argon2(profile);
    let mut digest = [0u8; ARGON2_HASH_LEN];
    argon2
        .hash_password_into(input, ARGON2_SALT, &mut digest)
        .expect("Argon2 hash failed");
    digest
}

/// Run iterative Argon2d hashing (bulk, no progress).
///
/// - `input`: 8 bytes (64 bits of stage-1 entropy)
/// - `gui_iterations`: number of hash-then-feed-back cycles.
///   1 means hash once; N means hash N times, each feeding the previous
///   digest back as input.
/// - Returns 32 bytes (256-bit final digest)
pub fn iterative_argon2(input: &[u8; 8], gui_iterations: u32, profile: Profile) -> [u8; ARGON2_HASH_LEN] {
    let argon2 = make_argon2(profile);

    // First iteration: hash the 8-byte input
    let mut digest = [0u8; ARGON2_HASH_LEN];
    argon2
        .hash_password_into(input, ARGON2_SALT, &mut digest)
        .expect("Argon2 hash failed");

    // Subsequent iterations: feed digest back as input
    for _ in 1..gui_iterations {
        let prev = digest;
        argon2
            .hash_password_into(&prev, ARGON2_SALT, &mut digest)
            .expect("Argon2 hash failed");
    }

    digest
}
