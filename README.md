# Great Wall Reference Implementation

Bijective mapping between BIP39 mnemonic seeds and Burning Ship fractal
locations, with an Argon2-based chained pipeline: one 32-bit point per
stage, each stage its own fractal derived by hashing all preceding points.

> **Design documentation** for the encoder lives in the `great-wall-docs`
> repository (`great-wall-core/DESIGN.md`) — the single source of truth. This
> README is the only doc kept in this repo.
>
> **Versioning guarantee.** This core implements **chained protocol
> `PROTOCOL_VERSION = 0.2.0`** (see `burning_ship/protocol.py`), and the
> authoritative `DESIGN.md` declares the *same* version. The two are therefore
> verifiably in sync: bump the protocol version in both — and re-stamp the
> encode/decode JSON `protocol_version` field — whenever the protocol's
> behaviour changes. (This is independent of the Rust `ENGINE_VERSION`, the
> single-fractal encode/decode algorithm, which is unchanged at `0.1.0`.)
>
> **Pre-1.0 / test-vector policy.** The protocol is pre-`1.0.0` (unstable;
> more changes are expected — new parameter families, etc.). Comprehensive
> frozen test vectors are intentionally **deferred to the stable `1.0.0`**
> release rather than rebuilt for every interim bump. This is safe because the
> test harness carries a **version guard**: any vector whose `protocol_version`
> differs from the current one is reported **STALE** (skipped, never counted as
> a pass), so stale vectors can never show false-green.

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT License](LICENSE-MIT) at your option.

## Why Great Wall?

Protecting a Bitcoin seed phrase is a problem with no good conventional
solution. Every existing approach sacrifices at least one desirable
property. Great Wall provides all four at once:

1. **Knowledge-Based Authentication.** Your secret lives entirely in your
   memory — no device, physical vault, or geographic location required.
2. **Individual Custody.** You depend on no one else. The core premise of
   Bitcoin — full self-custody — is kept intact.
3. **Non-Obscurity.** The method is not a secret trick that fails the
   moment an attacker learns about it. Nor does it rely on convincing the
   attacker that the stash doesn't exist or is smaller than it really is.
4. **Coercion-Resistance.** The threat of violence is ineffective as a
   means to obtain the secret leading to the stash.

> **In one sentence:** it's all in your head (1), in nobody else's (2),
> the attacker is aware of that (3), and is nevertheless unable to rob
> it (4).

This sounds like having your pie and eating it too — many times over. It
is only possible because the secret knowledge (that is only in your head)
is *tacit*, and the interface for *deploying* it (to convert it into a key)
is gated by an inescapably lengthy computation.

### How it works (the simple version)

The Burning Ship fractals are indescribable labyrinths. Great Wall converts
your secret into exact coordinates on these labyrinths.

Your memory of *where* those points are is **tacit knowledge** — the same
kind of knowledge that lets you recognize a friend's face but not describe
it precisely enough for a stranger to pick them out of a crowd. You can
identify your locations by looking at the fractal, but you literally
cannot dictate them to someone else. There is no verbal shortcut: the
only way to extract the secret is to sit in front of the fractal and
point.

Great Wall splits your secret into one 32-bit point per stage. The first
point is encoded on the standard (canonical) fractal. But before *each*
later point can be encoded *or decoded*, the system requires a long,
memory-intensive computation using Argon2 — a key-stretching algorithm
where each step depends on the previous one (strictly sequential) and
requires gigabytes of RAM that cannot be traded for speed. The delay is
configurable: hours, days, or weeks. Each computation produces a unique key
that reshapes the fractal itself — generating a *new, private* labyrinth —
and its input is **all the preceding points**, so the labyrinths form a
chain: stage *k+1*'s fractal cannot even be derived until stage *k*'s point
is fixed.

Each next-stage fractal **does not exist** until its computation finishes.
No one — not even a fully cooperative user — can reveal the later locations
any faster, because there is nothing to point at until Argon2 is done, and
the chain must be walked in order.

This is why coercion fails: the user **cannot verbalize** their fractal
locations (tacit knowledge), and the later-stage fractals **cannot be
materialized** without completing the full Argon2 chain. Coercing the user
into giving up the entire secret effectively requires kidnapping them for at
least as long as the total Argon2 delay — and that delay compounds across the
chain (one memory-hard derivation per secret stage). That is the wall.

## Burning Ship Seed Encoder (`burning_ship/`)

Bijective mapping between BIP39 mnemonic seeds and locations in the
Burning Ship fractal, using I4F60 fixed-point arithmetic (60 fractional
bits, range [-8, +8)).

Because the protocol uses exactly one 32-bit point per stage,
`stages = entropy_bits / 32 = words / 3`, so **every** BIP39 size that is a
multiple of 32 bits is supported uniformly — one extra stage per extra 32 bits
(3 words). All sizes from 32 to a hard cap of **256 bits** are offered:

| Words | Entropy | Stages | Secret fractals | Tier |
|------:|--------:|-------:|----------------:|------|
| 3     | 32      | 1      | 0               | sub-standard |
| 6     | 64      | 2      | 1               | sub-standard |
| 9     | 96      | 3      | 2               | sub-standard |
| 12    | 128     | 4      | 3               | standard (default) |
| 15    | 160     | 5      | 4               | standard |
| 18    | 192     | 6      | 5               | standard |
| 21    | 224     | 7      | 6               | standard |
| 24    | 256     | 8      | 7               | standard |

**Hard cap at 256 bits / 24 words.** Larger mnemonics are valid BIP39 in
principle but are deliberately *not* offered: one more stage is the same
marginal mental effort for diminishing returns, so the better lever past 24
words is **more between-stage Argon2 iterations**, not more stages. (A future
"advanced pepper" field — pepper = a prior setup's result — could chain
multiple setups for anyone with a specific reason to exceed 256 bits; see the
design docs' next-steps.)

**Single Argon2 iteration count per setup, calibrated on-device.** To avoid
parameter explosion, one iteration count `N` is fixed at setup and applied to
**every** stage. `N` is the durable parameter; wall-clock "hours" is only a
*perishable label* on it — recovery reproduces the digest from `N`, so hardware
progress changes how long a given `N` takes, never correctness or security. The
interface offers a few **target durations** and calibrates `N` on the user's own
device. `N` is the **user's responsibility to memorize** (for hard recovery if
the device is lost); the protocol gracefully stores the sequence of intermediate
results so an approximate memory of `N` suffices (recognize the correct one).
Users are encouraged to set the time **conservatively (≈2×)** and then use the
TLP / jade-clock layer (see the `great-wallet` family) to tune the effective
per-session delay. Official policy; may be revisited.

**A note on the 32-bit (3-word, single-stage) mode.** With only one point it
has only the first (canonical) stage and no memory-hard chain, so its
coercion-resistance is *minimal but still nonzero*: a wrench attacker who lacks
a Great Wall–compatible app still fails, and the setup outperforms the
brute-force resistance of a 9-decimal-digit PIN. It is offered for completeness;
128 bits (12 words) remains the recommended default.

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
| `T`                      | Cycle active stage (through derived stages) |
| `V`                      | Toggle area visualization (bisection rectangles) |
| `<` / `>`                | Navigate bisection steps (when V active) |
| `W`                      | Cycle size: any of 3–24 words (1–8 stages, 32–256 bits) |
| `M`                      | Toggle manual bit-input mode        |
| `O` / `I`                | Enter bit 0 / bit 1 (manual mode)   |
| `Backspace`              | Undo last bit (manual mode)         |
| `X`                      | Stretch correction mode (click P1, P2) |
| `Z`                      | Clear stretch corrections           |
| `F2`                     | Random encode (full chain: all stages + Argon2 between them) |
| `F5`                     | Load session from JSON              |
| `F6`                     | Save session to JSON                |
| `Escape` / `Q`           | Quit                                |
| `Ctrl+C` / `Ctrl+V`     | Copy / Paste in text fields         |

### Chained Pipeline (one point per stage)

The entropy is split into `n_stages = entropy_bits / 32` chunks of 32 bits,
one point per stage.

**Stage 0** encodes its point using the canonical Burning Ship formula
(o=0, p=0, q=0).

**Stage k ≥ 1** encodes its point using a perturbed formula whose parameters
(o, p, q) are derived by hashing **all preceding points** with Argon2
(configurable profile and iteration count) then SHA-256 — a different fractal
with a different area tree for every stage. Because each derivation consumes
the whole prefix, the stages form a strict chain, and the honest derivation
cost is one memory-hard Argon2 run per secret stage (`n_stages − 1` total).

The chained pipeline lives in `burning_ship/protocol.py`
(`encode_entropy` / `decode_entropy` / `stage_params`); the CLI and viewer
both drive encode/decode through it.

### Argon2 Intermediate-State Checkpoints

Next to the Argon2 *Hash* button there is a **save intermediate**
checkbox. It is **off by default**.

- **Off (default):** the iterative Argon2 derivation is run from scratch
  and only the final digest is kept in memory. Nothing is written to
  disk. This preserves the wall-clock barrier — interrupting the
  computation forces a full restart, exactly as intended.
- **On:** every iteration's digest is appended to a checkpoint file
  (`.argon2_checkpoint_{input_hex}_{profile}.bin`) next to
  `argon2_pipeline.py`. On a subsequent run with the same input and
  profile, the highest saved iteration ≤ the target is used as the
  starting point. This is convenient for resuming after interruption,
  for trying nearby iteration counts when the exact target is
  forgotten, and for development.

> **Security note — your responsibility:** an intermediate-state file
> is precisely the artifact that lets an attacker skip the time cost
> Argon2 was meant to impose. **It is the user's responsibility to
> delete intermediate-state files (securely — e.g. `shred`, `srm`, or
> equivalent) once they have served their purpose.** Leaving them on
> disk defeats the entire point of the derivation. The core viewer
> deliberately does *not* auto-delete: it cannot know when "their
> purpose has been fulfilled" from the user's perspective, and a wrong
> guess would either delete state still needed for resumption or leave
> a forgotten file behind.

> **Sibling repositories** in the Great Wall family will abstract this
> complexity away from the UX. They wrap the core engine with
> Time-Lock-Puzzle (TLP) cryptographic gating of intermediate states
> and automatic secure deletion of intermediate derivation states once
> they are no longer needed — so end users don't have to manage
> checkpoint hygiene by hand. This core repository intentionally
> exposes the raw mechanism: it is the substrate those sibling tools
> build on, and surfaces the trade-off explicitly so it is impossible
> to forget.

### Manual Bit-Input Mode

Press `M` to enter manual mode. Use `O` (bit 0) and `I` (bit 1) to
manually bisect the area tree one level at a time. The area visualization
updates incrementally (O(1) per keypress via inherited seeds across FFI).
The stage's point auto-commits at 32 bits; you then run the Argon2 step to
unlock the next stage's fractal and continue.

### Point Selection & Validation

Press `S` to enter selection mode, then click points on the fractal.
Each clicked point is validated through the full bisection with
contraction — points that fall in contracted-away dead zones are
rejected with a status bar message.

Selection is **slot-based**: each click fills the *active slot*
(highlighted with a `←` next to its marker), and the active slot then
advances to the next still-empty slot. This means a wrong click is not
punitive — you can:

- Press `N` to cycle the active slot, and
- Click again on a slot you already filled to **overwrite** it.

Each stage holds a single point; once it is validly placed, its 32 bits are
decoded automatically, the next stage's fractal is derived from the cumulative
decoded bits, and selection advances to that stage. After the last stage, the
full BIP39 mnemonic is assembled.

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
  protocol.py            Chained one-point-per-stage encode/decode pipeline
  constants.py           All configuration, colors, size presets
  palettes.py            Color schemes and escape-count transforms
  encoding.py            BIP39 ↔ fractal encode/decode primitives
  argon2_pipeline.py     Argon2 hashing/chain, checkpoints, F2 pipeline
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
per point**. Since the protocol uses exactly one point per stage, a 12-word
BIP39 seed (128 entropy bits) uses 4 stages and a 24-word seed (256 bits)
uses 8 stages — one 32-bit point each.
