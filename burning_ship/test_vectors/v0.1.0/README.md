# Test Vectors v0.1.0

Engine version: `0.1.0`

All vectors use Argon2 profile `b` (basic, 1 GiB, t=32, p=1).

## Vectors

| File | Entropy | Mode | Argon2 iters | Purpose |
|------|---------|------|-------------|---------|
| `default_zeros_iter0.json` | 128-bit all zeros | default | 0 (identity) | Boundary: p=0 baseline |
| `default_ones_iter0.json` | 128-bit all ones | default | 0 | Opposite extreme |
| `default_single_bit_iter0.json` | 128-bit, only MSB set | default | 0 | Single bit difference |
| `vanity1_iter0.json` | "never use this word..." | default | 0 | Self-disclaiming BIP39 |
| `vanity2_iter0.json` | "never use this example..." | default | 0 | Self-disclaiming BIP39 |
| `vanity2_iter1.json` | "never use this example..." | default | 1 | Tests 1 Argon2 pass |
| `vanity2_iter2.json` | "never use this example..." | default | 2 | Tests Argon2 chaining + intermediates |
| `vanity3_iter0.json` | "test only example nut..." (case) | default | 0 | Self-disclaiming BIP39 |
| `vanity4_iter0.json` | "test only example nut..." (life) | default | 0 | Self-disclaiming BIP39 |
| `mini_zeros_iter0.json` | 64-bit all zeros | mini | 0 | Mini preset (6 words) |
| `mini_abandon_iter0.json` | "abandon...able" (6 words) | mini | 0 | Cross-mode test |
| `large_zeros_iter0.json` | 256-bit all zeros | large | 0 | Large preset (24 words) |
| `large_abandon_iter0.json` | "abandon...art" (24 words) | large | 0 | Cross-mode test |
| `default_abandon_iter0.json` | "abandon...about" (12 words) | default | 0 | Cross-mode test |

## Cross-mode invariant

The first point's leaf (stage 1, point 1) is identical across
`mini_abandon_iter0`, `default_abandon_iter0`, and `large_abandon_iter0`
because all three share the same first 32 bits of entropy and the area
tree is an invariant property of the fractal.

## Running

```bash
cd burning_ship
python3 test_vectors.py              # all tests
python3 test_vectors.py --verbose    # show diff details on failure
```
