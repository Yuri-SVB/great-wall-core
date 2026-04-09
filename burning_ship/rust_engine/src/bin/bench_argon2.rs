/// Argon2d benchmark for Great Wall profile tuning.
///
/// Measures wall-clock time per pass at various (m, p, t) combinations.
/// Designed to run on the target device for each tier:
///
///   Basic tier:    run on Android phone (Termux) or iOS
///   Advanced tier: run on desktop
///   Great Wall:    run on server (AWS r6i.4xlarge or similar)
///
/// # Running on Android (Termux)
///
///   pkg install rust
///   cd burning_ship/rust_engine
///   cargo run --release --bin bench_argon2 -- --tier basic
///
/// # Running on desktop
///
///   cargo run --release --bin bench_argon2 -- --tier advanced
///
/// # Running on server (e.g. AWS r6i.4xlarge with 128 GB)
///
///   cargo run --release --bin bench_argon2 -- --tier greatwall
///
/// # Running all tiers (if machine has enough RAM)
///
///   cargo run --release --bin bench_argon2

use argon2::{Algorithm, Argon2, Params, Version};
use std::time::Instant;

const HASH_LEN: usize = 32;
const SALT: &[u8] = b"greatwall";
const INPUT: &[u8] = b"\x01\x02\x03\x04\x05\x06\x07\x08";

fn bench_one(label: &str, mem_kib: u32, t: u32, p: u32) -> f64 {
    let params = Params::new(mem_kib, t, p, Some(HASH_LEN))
        .expect("valid params");
    let argon2 = Argon2::new(Algorithm::Argon2d, Version::V0x13, params);
    let mut digest = [0u8; HASH_LEN];

    let start = Instant::now();
    argon2.hash_password_into(INPUT, SALT, &mut digest).expect("hash failed");
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();

    println!(
        "  {:<32} m={:>6} MiB  t={}  p={}  → {:>7.3}s",
        label, mem_kib / 1024, t, p, secs,
    );
    secs
}

fn warmup(mem_kib: u32) {
    let params = Params::new(mem_kib, 1, 1, Some(HASH_LEN)).expect("valid params");
    let argon2 = Argon2::new(Algorithm::Argon2d, Version::V0x13, params);
    let mut digest = [0u8; HASH_LEN];
    argon2.hash_password_into(INPUT, SALT, &mut digest).expect("warmup failed");
}

fn bench_basic() {
    let m = 1_048_576; // 1 GiB
    println!("=== BASIC TIER (1 GiB) — run this on your phone ===\n");

    println!("Warming up (faulting in 1 GiB pages)...");
    warmup(m);
    println!();

    println!("--- Effect of p (parallelism) at t=1 ---");
    println!("  (Lower is better. p should match phone's big-core count.)");
    let mut times_p = Vec::new();
    for p in [1, 2, 4] {
        let t = bench_one(&format!("p={p}"), m, 1, p);
        times_p.push((p, t));
    }
    println!();

    println!("--- Effect of t (time cost) at p=2 ---");
    println!("  (Target: ~10s per pass on this device.)");
    let mut times_t = Vec::new();
    for t in [1, 2, 3, 4, 5, 6, 8, 10, 12] {
        let secs = bench_one(&format!("t={t}"), m, t, 2);
        times_t.push((t, secs));
        if secs > 20.0 {
            println!("  (stopping — already >20s)");
            break;
        }
    }
    println!();

    println!("--- Effect of t (time cost) at p=1 ---");
    println!("  (Compare: does p=1 help the defender?)");
    for t in [1, 2, 3, 4, 5, 6, 8] {
        let secs = bench_one(&format!("t={t}"), m, t, 1);
        if secs > 20.0 {
            println!("  (stopping — already >20s)");
            break;
        }
    }
    println!();

    // Summary
    println!("--- Summary ---");
    println!("  Best p for this device:");
    let best_p = times_p.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!("    p={} ({:.3}s at t=1)", best_p.0, best_p.1);
    println!();
    println!("  t for ~10s target (at p=2):");
    for (t, secs) in &times_t {
        if *secs >= 8.0 && *secs <= 15.0 {
            println!("    t={} → {:.3}s  ← good candidate", t, secs);
        }
    }
    // Interpolate if needed
    for window in times_t.windows(2) {
        let (t1, s1) = window[0];
        let (t2, s2) = window[1];
        if s1 < 10.0 && s2 > 10.0 {
            let t_interp = t1 as f64 + (10.0 - s1) / (s2 - s1) * (t2 - t1) as f64;
            println!("    Interpolated: t≈{:.1} for exactly 10s", t_interp);
        }
    }
}

fn bench_advanced() {
    let m = 33_554_432; // 32 GiB
    println!("=== ADVANCED TIER (32 GiB) — run this on your desktop ===\n");

    // Check if we have enough RAM
    println!("Attempting 32 GiB allocation...");
    println!("(If this OOMs, your machine cannot run Advanced tier.)\n");

    println!("--- Effect of p at t=1, 32 GiB ---");
    for p in [1, 2, 4, 8] {
        bench_one(&format!("p={p}"), m, 1, p);
    }
}

fn bench_greatwall() {
    let m = 134_217_728; // 128 GiB
    println!("=== GREAT WALL TIER (128 GiB) — run this on your server ===\n");

    println!("Attempting 128 GiB allocation...");
    println!("(Requires ~192 GB system RAM.)\n");

    println!("--- Effect of p at t=1, 128 GiB ---");
    for p in [1, 2, 4, 8] {
        bench_one(&format!("p={p}"), m, 1, p);
    }
}

fn bench_local() {
    // Auto-detect what we can run on this machine
    println!("=== LOCAL BENCHMARK (auto-scaled) ===\n");

    let m_1g = 1_048_576;
    println!("Warming up...");
    warmup(m_1g);
    println!();

    println!("--- 1 GiB, varying p (t=1) ---");
    for p in [1, 2, 4, 8] {
        bench_one(&format!("p={p}"), m_1g, 1, p);
    }
    println!();

    println!("--- 1 GiB, varying t (p=2) ---");
    for t in [1, 2, 3, 4, 6, 8] {
        bench_one(&format!("t={t}"), m_1g, t, 2);
    }
    println!();

    // Try 4 GiB as Advanced proxy
    let m_4g = 4_194_304;
    println!("--- 4 GiB (Advanced proxy, ÷8), varying p (t=1) ---");
    warmup(m_4g);
    for p in [1, 2, 4, 8] {
        bench_one(&format!("p={p}"), m_4g, 1, p);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tier = args.iter()
        .position(|a| a == "--tier")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    println!("Argon2d Great Wall Benchmark");
    println!("CPU cores: {}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

    // Try to detect RAM (Linux)
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        if let Some(line) = meminfo.lines().find(|l| l.starts_with("MemTotal:")) {
            println!("RAM: {}", line.trim_start_matches("MemTotal:").trim());
        }
    }
    println!();

    match tier {
        Some("basic") => bench_basic(),
        Some("advanced") => bench_advanced(),
        Some("greatwall") => bench_greatwall(),
        Some(other) => {
            eprintln!("Unknown tier: {}. Use: basic, advanced, greatwall", other);
            std::process::exit(1);
        }
        None => bench_local(),
    }
}
