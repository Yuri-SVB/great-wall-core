# Great Wall Reference Implementation

Bijective mapping between BIP39 mnemonic seeds and Burning Ship fractal
locations, with Argon2-based two-stage pipeline.

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT License](LICENSE-MIT) at your option.

## Burning Ship Seed Encoder (`burning_ship/`)

Bijective mapping between BIP39 mnemonic seeds and locations in the
Burning Ship fractal, using I4F60 fixed-point arithmetic (60 fractional
bits, range [-8, +8)).

Supports three size presets:

| Preset  | BIP39 Words | Entropy Bits | Points/Stage |
|---------|-------------|-------------|-------------|
| mini    | 6           | 64          | 1           |
| default | 12          | 128         | 2           |
| large   | 24          | 256         | 4           |

### Prerequisites

- Rust 1.70+ (with cargo)
- Python 3.8+
- numpy >= 1.24
- pygame >= 2.5

### Build

```bash
cd burning_ship/rust_engine

# Release build (no logs — default)
cargo build --release

# Release build with verbose logging to stderr
cargo build --release --features verbose
```

This produces `burning_ship/rust_engine/target/release/libburning_ship_engine.so`.

### Run

```bash
cd burning_ship
python3 viewer.py

# Optional: enable render cache (recommended for smoother panning)
python3 viewer.py --cache-size 1048576
```

### Run tests

```bash
cd burning_ship

# All tests in one go (Rust + bijection + frozen vectors + round-trips + meta tests)
bash run_tests.sh
```

Or individually:

```bash
cd burning_ship

# Rust unit tests
cd rust_engine && cargo test && cd ..

# Bijection test (1-8 bits)
python3 test_bijection.py

# Frozen vectors, round-trips, cross-mode, and meta tests
python3 test_vectors.py

# Single vector (useful for debugging)
python3 test_vectors.py --vector test_vectors/v0.1.0/vanity2_iter0.json

# Verbose output (show diffs on failure)
python3 test_vectors.py --verbose
```

### Controls

| Input                    | Action                              |
|--------------------------|-------------------------------------|
| Mouse wheel / `+` `-`   | Zoom in/out at cursor               |
| Arrow keys / drag        | Pan                                 |
| `R`                      | Reset view to origin                |
| `1`-`5`                  | Switch color scheme (Classic/Fire/Ice/Rainbow/HiCon) |
| `P`                      | Cycle escape-count transform (Identity/Square/Cube/Exp/Sqrt/Cbrt/Log) |
| `L`                      | Toggle brightness falloff (sigmoid cave-like dimming) |
| `Tab`                    | Focus BIP39 text input              |
| `Enter`                  | Encode BIP39 seed (when input focused) |
| `C`                      | Clear all encoded/decoded points    |
| `D`                      | Toggle debug mode (show hex fields, verbose info) |
| `S`                      | Toggle point-selection mode (click to decode) |
| `T`                      | Toggle Stage 1 / Stage 2            |
| `V`                      | Toggle area visualization (bisection rectangles) |
| `<` / `>`                | Navigate bisection steps (when V active) |
| `W`                      | Cycle size preset (mini/default/large) |
| `M`                      | Toggle manual bit-input mode        |
| `O` / `I`                | Enter bit 0 / bit 1 (manual mode)   |
| `Backspace`              | Undo last bit (manual mode)         |
| `X`                      | Stretch correction mode (click P1, P2) |
| `Z`                      | Clear stretch corrections           |
| `F2`                     | Random encode (stage 1 + Argon2 + stage 2) |
| `F5`                     | Load session from JSON              |
| `F6`                     | Save session to JSON                |
| `Escape` / `Q`           | Quit                                |
| `Ctrl+C` / `Ctrl+V`     | Copy / Paste in text fields         |

### Two-Stage Pipeline

**Stage 1** encodes the first half of entropy bits using the canonical
Burning Ship formula (o=0, p=0, q=0). The encoded bits are then hashed
with Argon2 (configurable profile and iteration count) to derive
stage-2 fractal parameters (o, p, q).

**Stage 2** encodes the remaining entropy bits using the perturbed
formula with the Argon2-derived parameters, producing a different
fractal with a different area tree.

### Manual Bit-Input Mode

Press `M` to enter manual mode. Use `O` (bit 0) and `I` (bit 1) to
manually bisect the area tree one level at a time. The area visualization
updates incrementally (O(1) per keypress via inherited seeds across FFI).
Points auto-commit at 32 bits and advance to the next point.

### Point Selection & Validation

Press `S` to enter selection mode, then click points on the fractal.
Each clicked point is validated through the full bisection with
contraction — points that fall in contracted-away dead zones are
rejected with a status bar message. Valid points are decoded to bits.

### Salt & SHA512

When a BIP39 mnemonic is decoded, a salt input field and SHA512 button
appear. Enter a salt string and click SHA512 to compute
`SHA-512(mnemonic + salt)` — the hex digest is copied to the clipboard.

### Session Save/Load

Press `F6` to save the current session (encoded points, Argon2
parameters, mnemonic) to a JSON file. Press `F5` to load. The size
preset is auto-detected from the entropy bit count.

### Project Structure

```
burning_ship/
  viewer.py              Main application (event loop, rendering, panel)
  constants.py           All configuration, colors, size presets
  palettes.py            Color schemes and escape-count transforms
  encoding.py            BIP39 ↔ fractal encode/decode
  argon2_pipeline.py     Argon2 hashing, checkpoints, F2 pipeline
  session.py             Clipboard helpers, session save/load
  text_input.py          BIP39 text field keyboard handling
  manual_mode.py         Manual O/I bit input with incremental encode
  bip39.py               BIP39 mnemonic ↔ bit conversion (6/12/24 words)
  burning_ship_engine.py Python ctypes bridge to Rust
  rust_engine/
    src/
      lib.rs             Crate root, log_verbose! macro
      fixed.rs           I4F60 fixed-point type
      fractal.rs         Burning Ship iteration
      discovery.rs       Island discovery (sampling + flood fill)
      bisect.rs          Bisection tree encode/decode
      render_cache.rs    FIFO cache for rendering
      argon2_hash.rs     Argon2 hashing via Rust argon2 crate
      ffi.rs             C-ABI exports for Python bridge
```

### Python API

```python
from burning_ship_engine import encode, decode, DiscoveryParams, Rect

bits = [1, 0, 1, 1, 0, 0, 1, 0]
result = encode(bits)
decoded = decode(result.point_re_raw, result.point_im_raw, len(bits))
assert bits == decoded
```

### Precision Limits

With I4F60 (60 fractional bits), the encoder reliably handles **32 bits
per point**. For a 12-word BIP39 seed (128 entropy bits), use 2 stages
of 2 points each (32 bits per point). For 24-word (256 bits), use
2 stages of 4 points.
