"""Cross-language smoke test for the orbit-protocol (0.5.0) FFI.

Verifies the Python bridge (burning_ship_engine.py) against the built Rust
engine's `bs_orbit_root` / `bs_theta` / `bs_orbit_commitment` /
`bs_master_secret` / `bs_export_key` / `bs_shamir_interp`, cross-checked with
`hashlib` (for H = SHA-512 and its BIP-340-style tagged hash TH) and a tiny
independent GF(2^32) reference (for Shamir).  Requires `cargo build --release`
in rust_engine/ first.

Run:  python3 test_orbit_ffi.py   (from burning_ship/)

`orbit_advance` is not run here (it runs real Argon2d at >= 1 GiB); its
orchestration is covered by the Rust `orbit_step_with` tests. Only its cheap
rejection of steps == 0 is exercised.
"""

import hashlib
import random

import burning_ship_engine as eng


# --- independent H / TH reference (the spec, not the engine) ---
ORBIT_POINT_BYTES = 64  # SHA-512
TAG_MASTER_SECRET = b"GreatWall/TGPO/master-secret/v1"
TAG_EXPORT_LABEL = b"GreatWall/TGPO/export-label/v1"


def ref_h(*parts):
    return hashlib.sha512(b"".join(parts)).digest()


def ref_th(tag, *parts):
    t = ref_h(tag)
    return ref_h(t, t, *parts)


# --- independent GF(2^32) reference (mirrors src/shamir.rs; reduction 0x8D) ---
_REDUCTION = 0x8D


def gf_mul(a, b):
    acc = 0
    while b:
        if b & 1:
            acc ^= a
        carry = a & 0x8000_0000
        a = (a << 1) & 0xFFFF_FFFF
        if carry:
            a ^= _REDUCTION
        b >>= 1
    return acc


def gf_eval(coeffs, x):
    acc = 0
    for c in reversed(coeffs):
        acc = gf_mul(acc, x) ^ c
    return acc


def primary_abscissa(k):
    return k + 1


def resistance_abscissa(k):
    return 0x8000_0000 | (k + 1)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_width():
    print("orbit point width (bs_orbit_point_len):")
    check(eng.ORBIT_POINT_BYTES == ORBIT_POINT_BYTES,
          f"engine publishes {ORBIT_POINT_BYTES}-byte orbit points (SHA-512)")


def test_orbit_root():
    print("orbit_root (o_0 = H(sigma) = SHA-512):")
    check(eng.orbit_root(b"abc") == ref_h(b"abc"),
          "orbit_root(b'abc') matches hashlib SHA-512")
    sigma = bytes(range(128))  # a 1024-bit salt, Namtso's width
    check(eng.orbit_root(sigma) == ref_h(sigma),
          "orbit_root over a 128-byte sigma matches hashlib")
    check(eng.orbit_root(sigma) != eng.orbit_root(sigma[:-1] + b"\x00"),
          "distinct sigma -> distinct root")


def test_theta():
    print("theta (theta_i_j = H(o_i || j)):")
    o = bytes([7]) * ORBIT_POINT_BYTES
    check(eng.theta(o, 0) == ref_h(o, (0).to_bytes(4, "big")),
          "theta(o, 0) matches H(o || j) with big-endian j")
    check(eng.theta(o, 3) != eng.theta(o, 4), "distinct board index -> distinct fractal")
    check(eng.theta(o, 0) != eng.theta(bytes([8]) * ORBIT_POINT_BYTES, 0),
          "distinct orbit point -> distinct fractal")


def test_commitment_and_master_secret():
    print("u_i / K_i (domain-separated):")
    o = bytes([1]) * ORBIT_POINT_BYTES
    sh = bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88])
    u = eng.orbit_commitment(o, sh)
    k = eng.master_secret(o, sh)
    check(u == ref_h(o, sh), "u_i matches H(o || sh)")
    check(k == ref_th(TAG_MASTER_SECRET, o, sh),
          "K_i matches TH(master-secret, o || sh)")
    check(u != k, "u_i != K_i (a leaked K_i does not advance the orbit)")
    check(eng.master_secret(o, sh) != eng.master_secret(o, sh[:-1] + b"\x89"),
          "distinct Sh -> distinct K_i")


def test_export_key():
    print("export_key (K_i^L = TH(export-label, K_i || L)):")
    k = eng.master_secret(bytes([2]) * ORBIT_POINT_BYTES, bytes([0x42] * 12))
    for label in (b"", b"EXAMPLE-1"):
        check(eng.export_key(k, label) == ref_th(TAG_EXPORT_LABEL, k, label),
              f"export_key(K_i, {label!r}) matches the reference")
    check(eng.export_key(k, b"") != k, "the empty label is derived too (K_i never leaves)")
    check(eng.export_key(k, b"A") != eng.export_key(k, b"B"),
          "distinct label -> distinct exported key")


def test_orbit_advance_rejects_zero_steps():
    print("orbit_advance (steps == 0 rejected, not aborted):")
    o = bytes(ORBIT_POINT_BYTES)
    try:
        eng.orbit_advance(o, bytes(8), 0)
    except ValueError:
        check(True, "steps == 0 raises ValueError")
    else:
        raise AssertionError("FAILED: steps == 0 must raise")


def test_shamir():
    print("shamir_interp (full Sh over GF(2^32), subset-invariant):")
    rng = random.Random(0xC0FFEE)
    for t in range(2, 6):
        s = t + 3  # t primary + 3 forgetting-resistance shares
        coeffs = [rng.randrange(1, 1 << 32) for _ in range(t)]
        xs = [primary_abscissa(k) for k in range(t)]
        xs += [resistance_abscissa(k) for k in range(s - t)]
        ys = [gf_eval(coeffs, x) for x in xs]

        # width: t points -> t coefficients
        sh0 = eng.shamir_interp(xs[:t], ys[:t])
        check(len(sh0) == t, f"t={t}: Sh has t coefficients ({t}*32 bits)")
        check(sh0 == coeffs, f"t={t}: recovers the original polynomial")

        # subset-invariance: several distinct t-subsets -> identical Sh
        subsets = [list(range(t)), list(range(1, t + 1)), list(range(s - t, s))]
        shs = []
        for sub in subsets:
            sx = [xs[i] for i in sub]
            sy = [ys[i] for i in sub]
            shs.append(eng.shamir_interp(sx, sy))
        check(all(sh == coeffs for sh in shs),
              f"t={t}: any t of {s} shares reconstruct the identical Sh")
        check(len({eng.sh_to_bytes(sh) for sh in shs}) == 1,
              f"t={t}: identical Sh wire bytes across subsets (K_i stable)")


def test_setup_tiers():
    print("setup tiers (canonical per-stage thresholds t_i):")
    check(eng.setup_tier_thresholds(1) == [2, 2], "Setup 1 -> [2, 2] (entry)")
    check(eng.setup_tier_thresholds(2) == [2, 3], "Setup 2 -> [2, 3] (standard)")
    check(eng.setup_tier_thresholds(3) == [2, 3, 3], "Setup 3 -> [2, 3, 3]")
    check(eng.setup_tier_thresholds(4) == [2, 3, 3, 3], "Setup 4 -> [2, 3, 3, 3]")
    check(eng.setup_tier_substandard(1) is True, "Setup 1 is substandard")
    check(all(eng.setup_tier_substandard(k) is False for k in range(2, 10)),
          "Setup >= 2 is standard")
    check(eng.setup_tier_thresholds(0) == [], "invalid level 0 -> []")
    for k in range(2, 10):
        deep = eng.setup_tier_thresholds(k)[1:]
        check(all(t > 2 for t in deep), f"Setup {k}: every deep r_i > 2")


def main():
    print(f"engine version: {eng.get_engine_version() if hasattr(eng, 'get_engine_version') else '?'}")
    test_width()
    test_orbit_root()
    test_theta()
    test_commitment_and_master_secret()
    test_export_key()
    test_orbit_advance_rejects_zero_steps()
    test_shamir()
    test_setup_tiers()
    print("\nALL ORBIT FFI SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
