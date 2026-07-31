"""Cross-language smoke test for the orbit-protocol (0.4.0) FFI.

Verifies the Python bridge (burning_ship_engine.py) against the built Rust
engine's `bs_orbit_root` / `bs_theta` / `bs_master_secret` / `bs_shamir_interp`,
cross-checked with `hashlib` (for H = SHA-256) and a tiny independent GF(2^32)
reference (for Shamir).  Requires `cargo build --release` in rust_engine/ first.

Run:  python3 test_orbit_ffi.py   (from burning_ship/)

`orbit_advance` is intentionally not exercised here (it runs real Argon2d at
>= 1 GiB); its orchestration is covered by the Rust `orbit_step_with` test and
its K_i out-param equals master_secret(o, sh), verified below.
"""

import hashlib
import random

import burning_ship_engine as eng


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


def test_orbit_root():
    print("orbit_root (o_0 = H(sigma) = SHA-256):")
    check(eng.orbit_root(b"abc") == hashlib.sha256(b"abc").digest(),
          "orbit_root(b'abc') matches hashlib SHA-256")
    sigma = bytes(range(128))  # a 1024-bit salt, Namtso's width
    check(eng.orbit_root(sigma) == hashlib.sha256(sigma).digest(),
          "orbit_root over a 128-byte sigma matches hashlib")
    check(eng.orbit_root(sigma) == eng.orbit_root(sigma), "deterministic")
    check(eng.orbit_root(sigma) != eng.orbit_root(sigma[:-1] + b"\x00"),
          "distinct sigma -> distinct root")


def test_theta():
    print("theta (theta_i_j = H(o_i || j)):")
    o = bytes([7]) * 32
    check(eng.theta(o, 0) == hashlib.sha256(o + (0).to_bytes(4, "big")).digest(),
          "theta(o, 0) matches H(o || j) with big-endian j")
    check(eng.theta(o, 3) != eng.theta(o, 4), "distinct board index -> distinct fractal")
    check(eng.theta(o, 0) != eng.theta(bytes([8]) * 32, 0),
          "distinct orbit point -> distinct fractal")


def test_master_secret():
    print("master_secret (K_i = H(o_i || Sh_i)):")
    o = bytes([1]) * 32
    sh = bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88])
    check(eng.master_secret(o, sh) == hashlib.sha256(o + sh).digest(),
          "master_secret matches H(o || sh)")
    check(eng.master_secret(o, sh) != eng.master_secret(o, sh[:-1] + b"\x89"),
          "distinct Sh -> distinct K_i")


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


def main():
    print(f"engine version: {eng.get_engine_version() if hasattr(eng, 'get_engine_version') else '?'}")
    test_orbit_root()
    test_theta()
    test_master_secret()
    test_shamir()
    print("\nALL ORBIT FFI SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
