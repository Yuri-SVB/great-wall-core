# Burning Ship Encoder — Design Document

## Purpose

The Burning Ship encoder implements a **bijective mapping** between arbitrary
bit sequences and points in the complex plane, using the structure of the
[Burning Ship fractal](https://en.wikipedia.org/wiki/Burning_Ship_fractal) as
a deterministic, content-dependent partition function.

Given a bit string **b** of length *n*, the encoder produces a complex point
**(c_re, c_im)** in I4F60 fixed-point representation such that decoding that
point with the same parameters recovers **b** exactly. The mapping is injective
(distinct bit strings yield distinct points) and the round-trip is lossless.

The intended application is encoding small secrets (e.g. portions of a BIP-39
mnemonic seed) into fractal coordinates that can later be decoded without any
stored state beyond the shared parameters.

---

## Architecture Overview

```
                ┌──────────────────────────────────┐
                │         Python FFI bridge         │
                │    (burning_ship_engine.py)        │
                └──────────┬───────────────────────┘
                           │ ctypes / cdylib
                ┌──────────▼───────────────────────┐
                │         Rust core engine          │
                │  ┌────────┐  ┌──────────────────┐ │
                │  │fixed.rs│  │   fractal.rs      │ │
                │  │ I4F60  │  │ escape_count()   │ │
                │  └────┬───┘  └────────┬─────────┘ │
                │       │               │            │
                │  ┌────▼───────────────▼─────────┐ │
                │  │       discovery.rs            │ │
                │  │  island finding (flood fill)  │ │
                │  └──────────────┬────────────────┘ │
                │                 │                   │
                │  ┌──────────────▼────────────────┐ │
                │  │         bisect.rs             │ │
                │  │  encode() / decode()          │ │
                │  └───────────────────────────────┘ │
                │                                    │
                │  ┌───────────────────────────────┐ │
                │  │           ffi.rs              │ │
                │  │   C-ABI exports (bs_*)        │ │
                │  └───────────────────────────────┘ │
                └──────────────────────────────────┘
```

**Modules:**

| Module           | Role                                              |
|------------------|----------------------------------------------------|
| `fixed.rs`       | I4F60 fixed-point type with checked arithmetic     |
| `fractal.rs`     | Burning Ship iteration with overflow escape         |
| `discovery.rs`   | Island discovery via random sampling + flood fill   |
| `bisect.rs`      | Bisection tree: encode and decode bit sequences     |
| `render_cache.rs`| FIFO hash table cache for escape_count (rendering)  |
| `ffi.rs`         | C-ABI interface consumed by the Python bridge       |

---

## No Floating-Point in Deterministic Paths

**Every computation that affects the encode/decode bijection is performed in
integer or fixed-point arithmetic.** IEEE 754 floating-point (`f64`) is used
*only* for display, debug output, and the pixel-grid rendering viewport.

No `f64` operation participates in:

- PRNG seeding or generation
- Island barycenter computation (accumulated as `i128`, divided as integer)
- Island weighting (integer inverse-count weights)
- Split coordinate computation (Fixed)
- Contraction factor computation (integer fraction `num/den`)
- Contraction application (`i128` multiply + divide)
- Child selection or point membership tests
- Initial area bounds (passed as raw `i64` Fixed across FFI)

The FFI boundary passes all encode/decode coordinates as `Fixed`
(`#[repr(transparent)]` newtype over `i64`), eliminating the earlier `f64`
conversion at entry.  The Python bridge stores `Rect` values as raw `i64`
and provides `_f64()` accessors for display only.

---

## Fixed-Point Arithmetic (I4F60)

All fractal computation uses a custom 64-bit signed fixed-point format called
**I4F60**:

```
 Bit 63 (sign)    Bits 62–60 (integer)    Bits 59–0 (fraction)
┌──┬──────────────┬──────────────────────────────────────────┐
│ S│  3 int bits  │            60 fractional bits            │
└──┴──────────────┴──────────────────────────────────────────┘
```

- **Total width:** 64 bits (stored as Rust `i64`)
- **Integer range:** [-8, +8)
- **Fractional precision:** 2^(-60) ≈ 8.67 × 10^(-19)
- **The constant ONE:** `1i64 << 60`

### Why fixed-point?

1. **Determinism.** IEEE 754 floating-point can produce platform-dependent
   results due to extended precision, FMA fusion, and rounding-mode
   differences. Fixed-point addition and subtraction are exact; multiplication
   uses a widening `i64 → i128` product followed by a right-shift, which is
   fully deterministic on any two's-complement architecture.

2. **Natural escape criterion.** Rather than testing `|z| > R` for some
   bailout radius, the engine treats *any arithmetic overflow* as escape.
   This is both simpler and more precise: it catches divergence the instant
   the orbit leaves the representable range.

### Checked operations

Every arithmetic step returns `Option<Fixed>`:

| Operation      | Implementation                                                |
|----------------|---------------------------------------------------------------|
| `checked_add`  | `i64::checked_add` — detects signed overflow                  |
| `checked_sub`  | `i64::checked_sub` — detects signed overflow                  |
| `checked_mul`  | Widen to `i128`, multiply, right-shift by 60, range-check     |
| `checked_abs`  | Returns `None` for `i64::MIN` (no positive counterpart)       |
| `checked_neg`  | `i64::checked_neg` — catches `i64::MIN`                       |
| `midpoint`     | `(a >> 1) + (b >> 1) + (a & b & 1)` — overflow-free          |

---

## The Burning Ship Fractal

The Burning Ship fractal is defined by the recurrence:

```
z₀ = 0

z_{n+1} = ( |Re(zₙ)| + i·|Im(zₙ)| )² + c
```

Expanding the squaring step:

```
Re(z_{n+1}) = |Re(zₙ)|²  −  |Im(zₙ)|²  +  Re(c)
Im(z_{n+1}) = 2·|Re(zₙ)|·|Im(zₙ)|      +  Im(c)
```

A point **c** is *in the set* if the orbit never escapes (i.e. never
overflows). The **escape count** is the iteration index at which the first
overflow occurs, or `None` if the point survives `max_iter` iterations.

### Computation sequence

For each iteration the engine performs these checked steps in order:

1. `abs_zr = |Re(z)|` — overflow if `z_re == i64::MIN`
2. `zr2 = abs_zr * abs_zr` — widening multiply + shift
3. `zi2 = z_im * z_im`
4. `diff = zr2 - zi2`
5. `new_re = diff + c_re`
6. `abs_zi = |Im(z)|` — overflow if `z_im == i64::MIN`
7. `prod = abs_zr * abs_zi`
8. `double_prod = prod * 2` — checked left-shift by 1
9. `new_im = double_prod + c_im`

If *any* step returns `None`, the point escapes at that iteration.

---

## Island Discovery

An **island** is a connected region of the complex plane where all points
share the same escape count. Islands capture the nested, self-similar
structure of the fractal boundary.

### Algorithm

1. **Seed the PRNG.** A SplitMix64 generator is deterministically seeded from
   the **bisection path string** (accumulated history of directional choices)
   plus a fixed extra seed value (`rng_seed`). This ensures that *the same
   path always produces the same island set*.

2. **Random sampling.** Draw random points uniformly within the rectangle.
   Skip points that collide with the stored flood-fill data (see step 7).

3. **Escape test.** Compute the escape count at each sample point. Skip
   points that are inside the set (escape count = `None`) or have low
   escape count (< 20). Empirically, the first meaningful islands only
   appear around escape count ~17; points below this threshold are
   trivially far from the fractal boundary and never form useful
   connected regions.

4. **Flood fill (BFS).** From each qualifying sample point, expand in
   4-connected directions at a fixed step size (`pixel_delta`). Only include
   neighbours with the *same* escape count. Neighbours with a **higher**
   escape count are treated as **boundary** (farther from the set — not
   expanded). Neighbours with a **lower** escape count are noted as
   candidates for redirection (see step 6). Track visited cells on a
   discretized grid (coordinates divided by `pixel_delta`) to avoid revisits.

5. **Adaptive resolution.** The initial pixel step size `p0` is the largest
   power-of-two such that `(width × height) / p0² ≥ min_grid_cells`. If
   the flood fill hits the point budget (`max_flood_points`) at the current
   resolution, the pixel size is doubled (one left-shift) and the fill
   retries. If the pixel size reaches the coarsest allowed
   (`min(width, height) >> p_max_shift`) and the budget is still exceeded,
   the region is marked "too large" and excluded.

6. **Lower-escape redirection.** If a flood fill encounters *any* neighbour
   whose escape count is lower than the target (but still ≥ 20), the fill
   restarts from that lower-escape point. Lower escape counts correspond to
   regions
   *closer* to the fractal boundary, which tend to form larger connected
   components (true islands) rather than isolated single-pixel noise.
   Conversely, higher-escape neighbours are treated strictly as boundary and
   never trigger redirection — they represent points farther from the set
   whose escape-count bands fragment into chaotic, unconnectable singletons
   at fine resolution.

7. **Collision store (FloodFillStore).** All flood-filled points are stored in
   a single lexicographically ordered `BTreeSet<(re, im, flood_id)>`. Each
   entry records the raw `i64` coordinates of a visited point together with the
   ID of the flood fill that produced it. Before starting a new flood fill from
   a random sample, the algorithm tests the sample point against this store
   using a **single vicinity check** at the coarsest pixel radius
   (`pixel_delta` at `p_max_shift`):

   - **Exact coincidence:** if any stored point has the same `(re, im)`, the
     sample is skipped immediately.
   - **Interior test:** the store scans all entries whose `re` lies within
     `[query_re − radius, query_re + radius]` and whose `im` lies within
     `[query_im − radius, query_im + radius]`.  For each stored point it
     records which cardinal direction it lies in relative to the query (left,
     right, above, below) using per-flood-ID bitflags.  If any single flood ID
     has points in **all four** cardinal directions, the query point is interior
     to that flood fill and the sample is skipped.

   This replaces the earlier two-BTreeMap design and multi-radius doubling loop
   with a single data structure and a single-pass range scan.

8. **Termination.** Stop when any of: `target_good` islands found,
   `exclusion_threshold_num / 256` of the total area excluded (tracked as
   precise pixel-count × pixel-delta² sums in `u128`), or `max_attempts`
   samples drawn.

### Island record

Each discovered island stores:

| Field          | Type       | Meaning                                           |
|----------------|------------|---------------------------------------------------|
| `center_re`    | `Fixed`    | Barycenter (integer mean of flood-filled points)  |
| `center_im`    | `Fixed`    |                                                   |
| `pixel_count`  | `u64`      | Number of flood-filled cells                      |
| `pixel_delta`  | `Fixed`    | Step size used during the flood fill               |
| `escape_count` | `u32`      | Shared escape count of all points                 |
| `bbox`         | `Rect`     | Bounding box of the flood-filled region           |

The barycenter is computed by accumulating coordinates in `i128` and dividing
by count — no floating-point involved.

### SplitMix64 PRNG

The PRNG used for all random sampling is SplitMix64, with wrapping arithmetic
on `u64`:

```
z ← z +w 0x9E3779B97F4A7C15         (golden-ratio increment, wrapping add)
z ← z ⊕ (z >> 30)
z ← z ×w 0xBF58476D1CE4E5B9         (wrapping multiply)
z ← z ⊕ (z >> 27)
z ← z ×w 0x94D049BB133111EB         (wrapping multiply)
z ← z ⊕ (z >> 31)
```

This is a well-known, fast, full-period 64-bit generator with good
statistical quality. Its determinism is critical: given the same seed, encode
and decode will discover *identical* island sets.

### PRNG seeding from path string

The PRNG is seeded from the **bisection path string** (not the rectangle
coordinates), combined with a fixed extra seed (`rng_seed`, default `0x42`).
Each byte of the path is folded into the state via XOR + SplitMix64:

```
h ← rng_seed
for each byte b in path_string:
    h ← h ⊕ b;  h ← splitmix(h)
state ← h
```

This makes the PRNG depend on the abstract bisection history rather than the
concrete coordinate values, which is more meaningful and stable across the
full pipeline (see "Path String" below).

---

## The Bisection Tree (Encoding)

The encoder maps a bit string to a point by recursively bisecting a rectangle,
using fractal structure to guide each split. For each input bit, the rectangle
is split in two and one half is selected.

### Step-by-step

For each bit *bᵢ* in the input:

1. **Discover islands** in the current rectangle (as described above).

2. **Compute the weighted median** of all discovered island barycenters along
   the split axis. Each island's weight is its *score*, computed as a
   fixed-point logarithm:

   ```
   scoreⱼ = log₂(good_total_area / flood_areaⱼ)     (fixed-point u128, 16 fractional bits)
   ```

   where `flood_areaⱼ = pixel_countⱼ × pixel_deltaⱼ²` and
   `good_total_area` is the sum over all good islands (all in `u128`).
   The `log₂` is computed via a bitwise algorithm (MSB position for the
   integer part, repeated squaring for fractional bits) — no floating-point.
   This compresses the dynamic range, preventing extremely small islands from
   dominating the median while preserving fine-grained discrimination between
   islands of similar size.

   The weighted median is the coordinate where the cumulative weight from
   below reaches half the total weight. All weights, sums, and comparisons
   are `u128` — no floating-point is involved. The island coordinates are
   `Fixed` (`i64`); sorting is by raw `i64` value. The result is a `Fixed`
   coordinate taken directly from an island barycenter.

3. **Choose the split axis:** the longer dimension of the current rectangle
   (vertical if width ≥ height, horizontal otherwise). Width and height are
   compared as raw `i64` differences.

4. **Clamp the barycenter** to within 10% of the edges. The margin is
   computed as `span / 10` (integer division) where `span` is the rectangle's
   extent along the split axis. This prevents degenerate near-empty children:
   without clamping, a barycenter very close to one edge could produce a child
   with near-zero area, leaving too little room for meaningful island
   discovery on subsequent levels. The 10% margin ensures the smaller child
   always gets at least 10% of the split dimension — small enough to allow
   genuinely asymmetric splits, but large enough to prevent pathological
   degeneracy.

5. **Split at the clamped barycenter.** Two child rectangles are created,
   sharing the non-split dimension.

6. **Identify children by span along the split axis.** The child whose span
   (in raw `i64` units) is larger is the "larger" child; the other is the
   "smaller" child. This comparison uses only integer subtraction.

### Directional bit semantics

Bits map to spatial direction, not relative size:

- **Vertical split** (split on re axis): `0 = left` (lo re), `1 = right` (hi re)
- **Horizontal split** (split on im axis): `0 = up` (hi im), `1 = down` (lo im)

The `chose_larger` flag is *derived* from which child the directional choice
lands on (does the chosen child happen to be the larger one?), rather than
driving the choice.  Contraction still applies when the larger child is
selected, regardless of which bit value triggered it.

7. **Consume bit *bᵢ*:**
   - `bᵢ = 0` → select the left (vertical) or up (horizontal) child
   - `bᵢ = 1` → select the right (vertical) or down (horizontal) child

8. **Contract (if larger selected).** When the chosen child happens to be the
   larger one, it is contracted toward the split line to prevent exponential
   area growth relative to the smaller child. The contraction factor is
   computed as an exact integer fraction:

   ```
   s = min(lo_span, hi_span)      (smaller child's span, i64)
   l = max(lo_span, hi_span)      (larger child's span, i64)

   numerator   = l + 3·s
   denominator = 4·l
   ```

   This implements `f(r) = (1 + 3r) / 4` where `r = s/l`, entirely in
   integer arithmetic. The contraction is applied as:

   ```
   new_edge_distance = old_edge_distance × numerator / denominator
   ```

   computed in `i128` to avoid overflow. When children are equal
   (`s = l`), `f = 1` (no contraction). When maximally unequal
   (`s → 0`), `f = 1/4` (contract to 25% of original span).

9. **Continue** with the contracted/selected child as the new rectangle.

After all bits are consumed, the **center of the final rectangle** is the
encoded point (computed via the overflow-free `midpoint` operation).

### Fallback

If island discovery finds no islands (the rectangle contains only set-interior
or trivially-escaping points), the encoder falls back to **geometric
bisection** at the rectangle's center (via `midpoint`), with no contraction.

---

## Path String

The bisection tree builds a **path string** that encodes the full history of
directional choices.  The string starts with `O` (origin) and appends one
character per bisection level:

| Split axis   | Direction | Smaller half | Larger half |
|-------------|-----------|:------------:|:-----------:|
| Vertical    | Left      | `l`          | `L`         |
| Vertical    | Right     | `r`          | `R`         |
| Horizontal  | Up        | `u`          | `U`         |
| Horizontal  | Down      | `d`          | `D`         |

Lowercase = the chosen half is the smaller one; UPPERCASE = the larger one.

### Path prefix per point

Each point's bisection starts from `path_prefix = "O"`, regardless of
preceding points. This makes the area tree an **invariant property of each
fractal** — determined solely by (o, p, q) — rather than depending on the
encoding pipeline order. Same entropy bits always produce the same area tree.

---

## Decoding

Decoding reproduces the *exact same* bisection tree — because the tree is
fully determined by the path prefix and the deterministic PRNG — and reads off
which half contains the query point at each level.

Encode and decode share a single implementation (`bisect_core`) that
parameterizes over whether bits come from input (encode) or are inferred from
point membership (decode).

For each bit position *i* from 0 to *n−1*:

1. Reproduce the island discovery, barycenter, and split (identical to
   encoding — same rectangle, same PRNG seed, same result).

2. Test which child contains the query point:
   - If the point coordinate along the split axis is `≥ split_coord`, the
     point is in `child_hi`; otherwise in `child_lo`.
   - Map to directional bit: for vertical splits, `hi = right = 1`,
     `lo = left = 0`.  For horizontal splits, `hi = up = 0`,
     `lo = down = 1`.

3. Apply the same contraction to the selected child.

4. Continue with the next level.

The decoded bit string is the sequence of 0/1 values recovered at each level.

### Encoded point representation

The encoded point is returned as a pair of **raw `i64` values** (the internal
`Fixed` representation). The decoder accepts raw `i64` values directly. This
avoids any `f64 → Fixed → f64` round-trip that could introduce
non-determinism. The Python bridge exposes both:

- `point_re`, `point_im`: `f64` values for display/visualization
- `point_re_raw`, `point_im_raw`: `i64` values for exact decode

### Why is this a bijection?

- **Determinism:** The PRNG is seeded from the path string (accumulated
  bisection history). Given the same path prefix and parameters, discovery
  always produces the same islands, the same weights, the same barycenter,
  and the same split. Encode and decode therefore walk the same tree.

- **No floating-point:** Every step from PRNG through barycenter, split,
  contraction, and membership test uses only integer/fixed-point arithmetic.
  The result is identical on any conforming two's-complement implementation.

- **Semi-open intervals:** All rectangles use the convention `[min, max)` —
  closed on the lower bound, open on the upper. `Rect::contains` tests
  `re >= re_min && re < re_max && im >= im_min && im < im_max`. At each
  split, the two children `[min, split)` and `[split, max)` partition
  the parent exactly with no overlap. The decode test `point >= split_coord`
  assigns boundary points unambiguously to the high child.

- **Injectivity:** Different bit strings select different paths through the
  tree. Different paths lead to non-overlapping final rectangles. The
  centers of non-overlapping rectangles are distinct points.

---

## Precision Limits

With I4F60 (60 fractional bits), the representable precision is ~10^(−18).
Each bisection level roughly halves one dimension of the rectangle, consuming
approximately 1 bit of positional precision. After *n* bisections, the
rectangle's smaller dimension has shrunk by a factor of roughly 2^n (modulated
by contraction).

In practice, the encoder reliably handles **32 bits per point.** Beyond that,
the rectangle becomes too small for island discovery to find meaningful
structure at the minimum pixel resolution.

Three size presets are supported (always 2 stages, 32 bits per point):

| Preset  | Points/Stage | Total Entropy | BIP39 Words |
|---------|-------------|--------------|-------------|
| mini    | 1           | 64 bits      | 6           |
| default | 2           | 128 bits     | 12          |
| large   | 4           | 256 bits     | 24          |

Stage 1 encodes the first half of entropy bits using the canonical Burning
Ship formula (o=0, p=0, q=0); Stage 2 encodes the remaining half using a
perturbed formula with Argon2-derived parameters (o, p, q).

---

## Default Initial Area

The default rectangle covers the **canonical Burning Ship fractal region**:

```
Re ∈ [−2.5, +1.5)      Im ∈ [−2.0, +1.5)
```

This 4.0 × 3.5 rectangle tightly contains the main body and principal
miniatures of the Burning Ship fractal, ensuring that random sampling
concentrates on structurally interesting regions rather than wasting
attempts on trivially-escaping points far from the set.

---

## Discovery Parameters

All parameters are integers — no floating-point values cross the API boundary.

| Parameter                | Rust default | GUI override | Meaning                                                        |
|--------------------------|-------------|-------------|----------------------------------------------------------------|
| `max_iter`               | 128         | 256         | Maximum fractal iterations per escape test                     |
| `min_grid_cells`         | 4,096       | 1,048,576   | p0 = largest power-of-2 s.t. (w×h)/p0² ≥ min_grid_cells       |
| `p_max_shift`            | 3           | 3           | Coarsest pixel size = `min(width,height) >> 3` (≈ 1/8)        |
| `max_flood_points`       | 50,000      | 10,000      | Point budget per flood fill                                    |
| `target_good`            | 100         | 64          | Stop after finding this many good islands                      |
| `exclusion_threshold_num`| 230         | 204         | Stop when `excluded_cells × 256 / total_cells ≥ threshold`    |
| `max_attempts`           | 500,000     | 500,000     | Maximum random sampling attempts                               |
| `rng_seed`               | 0x42        | 0x42        | Deterministic PRNG seed                                        |

The GUI uses faster, coarser parameters than the Rust defaults to keep encoding
interactive. The test suite uses its own set (`max_iter=200`, `target_good=30`,
`max_flood_points=10000`, `min_grid_cells=1024`, `p_max_shift=1`,
`exclusion_threshold_num=204`).

---

## FFI Interface

The Rust engine is compiled as a C dynamic library (`cdylib`). All exported
symbols use the `bs_` prefix and C calling convention (`extern "C"`).

| Function                       | Signature (abridged)                              | Purpose                                  |
|--------------------------------|---------------------------------------------------|------------------------------------------|
| `bs_escape_count`              | `(f64, f64, u32) → i32`                           | Single-point escape count (−1 = in set)  |
| `bs_render_viewport`           | `(origin, step, w, h, max_iter, *pixels)`          | Fill pixel buffer with escape map        |
| `bs_render_viewport_generic`   | `(origin, step, w, h, max_iter, p, *pixels)`       | Fill pixel buffer using perturbed formula|
| `bs_encode`                    | `(*bits, n, area, params, seed, p, prefix) → *H`   | Encode bits → opaque result handle       |
| `bs_encode_result_point`       | `(*H, *Fixed, *Fixed)`                             | Extract encoded point as Fixed           |
| `bs_encode_result_path`        | `(*H, *buf, buflen) → u32`                        | Extract path string (NUL-terminated)     |
| `bs_encode_result_bits`        | `(*H) → u32`                                      | Bits consumed                            |
| `bs_encode_result_num_steps`   | `(*H) → u32`                                      | Number of bisection steps                |
| `bs_encode_result_step`        | `(*H, idx, ...out pointers...)`                    | Per-step data (all Fixed/integer)        |
| `bs_encode_result_step_area`   | `(*H, idx, *Fixed×4)`                              | Area rectangle for a bisection step      |
| `bs_encode_result_final_rect`  | `(*H, *Fixed×4)`                                   | Final rectangle bounds                   |
| `bs_encode_result_num_seeds`   | `(*H) → u32`                                      | Inherited seed count for incremental enc |
| `bs_encode_result_seeds`       | `(*H, *buf, buflen)`                               | Copy seeds to buffer (24 bytes each)     |
| `bs_encode_with_seeds`         | `(*bits, n, area, params, ..., *seeds, nseed) → *H`| Encode with initial inherited seeds      |
| `bs_encode_result_free`        | `(*H)`                                             | Free handle memory                       |
| `bs_decode`                    | `(Fixed², n, area, params, seed, p, prefix, *out)` | Decode Fixed point → bit array           |
| `bs_decode_full`               | `(Fixed², n, area, params, seed, p, prefix, *bits, *rect, *valid)` | Decode + leaf rect + validity |
| `bs_get_precision`             | `(*u32, *u32)`                                     | Returns (60, 3) for I4F60               |
| `bs_cache_*`                   | various                                            | Render cache management (stage 1 & 2)   |
| `bs_argon2_hash`               | `(*u8, u32, u8, *u8)`                              | Iterative Argon2d (+ profile)            |
| `bs_argon2_single`             | `(*u8, u32, u8, *u8)`                              | Single Argon2d pass (+ profile)          |

**Key FFI changes from earlier versions:**

- All encode/decode area bounds and output coordinates are `Fixed`
  (`#[repr(transparent)]` over `i64`), not `f64`.
- `bs_encode_result_point` replaces the old `_point` (f64) and `_point_raw`
  (i64) pair — there is now only one accessor returning `Fixed`.
- `bs_encode`, `bs_decode`, `bs_decode_full` accept a **path prefix**
  (`*const u8` + length) for pipeline chaining.
- `bs_encode_result_path` is new: returns the accumulated path string.
- `bs_encode_result_step` returns split_coord as `Fixed`, contraction as
  separate `u64` numerator/denominator (not a single `f64` ratio).

---

## Determinism Contract

The round-trip guarantee depends on **exact reproducibility** across encode
and decode calls. The following invariants must hold:

1. The fixed-point format (I4F60) is identical: 60 fractional bits, `i64`
   storage, `i128` intermediate for multiplication.
2. The PRNG (SplitMix64) is seeded identically from the **path string**
   (byte-by-byte XOR + SplitMix64) with the same `rng_seed` value.
3. The island discovery algorithm (sampling order, flood fill BFS in
   4-connected order: +re, −re, +im, −im, collision store with single
   coarsest-radius vicinity test) is identical.
4. The weighted median uses fixed-point `u128` scores
   (`log2(good_total_area / flood_area)` with 16 fractional bits, computed
   via bitwise algorithm) for ordering. The resulting split coordinate is a
   `Fixed` value taken directly from an island barycenter. No floating-point
   is involved in score computation or median selection.
5. The contraction is the exact integer fraction `(l + 3s) / (4l)`, applied
   via `i128` multiply-then-divide.
6. The encoded point is stored and transmitted as raw `i64` values (not
   `f64`).  All area bounds cross the FFI as `Fixed` (`i64`).
7. The **path prefix** is always `"O"` for every point. The area tree is an
   invariant property of each fractal (determined by o, p, q alone).
8. **Inherited seeds** (flood-fill points from the previous bisection level)
   must be carried forward between levels. For incremental encoding, seeds
   are serialized across the FFI as 24-byte records (re: i64 LE, im: i64 LE,
   esc: u32 LE, pad: u32).

**No floating-point operation participates in any determinism-critical path.**

Any change to these components — even a different BFS neighbor order in the
flood fill — will break the bijection. This is by design: the fractal
structure serves as a shared, implicit codebook that both parties can
reconstruct independently.

---

## GUI Viewer: BIP39 Encode and Decode Workflows

The viewer (`viewer.py`) provides two complementary workflows: **encoding** a
BIP39 mnemonic into fractal points (1–4 per stage depending on preset), and
**decoding** user-selected fractal points back into entropy bits.

### Two-stage encoding architecture

The entropy is split into two equal halves, each encoded on a different
fractal surface:

| Stage | Formula                           | Parameters | Derived from        |
|-------|-----------------------------------|------------|---------------------|
| 1     | Canonical Burning Ship (d = 2)    | o=0, p=0, q=0 | —              |
| 2     | Perturbed BS with (o, p, q)       | Argon2-derived | SHA-256(Argon2(stage1 bits)) |

Stage 2's parameters (o, p, q) are derived from an Argon2d hash of the
Stage 1 entropy bits, making the second fractal surface dependent on the first.
The number of points per stage depends on the size preset (1, 2, or 4).

### Stage 1: Encoding (mnemonic → P1, P2)

Triggered by pressing **Enter** in the BIP39 input field, or clicking the
**Encode** button.

```
12 BIP39 words
    │
    ▼
mnemonic_to_bits()          12 words → 132 bits (128 entropy + 4 SHA-256 checksum)
    │                        Validates checksum; raises ValueError if invalid.
    ▼
entropy[:64]                 First 64 entropy bits → 2 chunks of 32 bits each.
    │                        (4-bit checksum is discarded; recomputed on decode.)
    ▼  (for each chunk)
encode(chunk, area, params)  Rust FFI: bisection tree over the canonical fractal.
    │                         Returns EncodeResult with:
    │                           • point_re, point_im     (f64, for display only)
    │                           • point_re_raw, point_im_raw  (i64, for lossless decode)
    │                           • bisection step data (for visualization)
    │                           • final_rect (leaf rectangle)
    ▼
P1, P2 displayed as crosshair markers on the fractal viewport.
```

### Argon2 transition (Stage 1 → Stage 2)

After encoding P1 and P2, the viewer derives the Stage 2 perturbation.
Three fixed profiles control the Argon2 parameters — the user chooses one at
setup time:

| Profile          | `m` (memory) | `p` (parallelism) | `t` (time cost) | Target device          |
|------------------|-------------:|--------------------|------------------|------------------------|
| **Basic**        |        1 GiB | 1                  | 2                | Mobile (3–4 GB+)       |
| **Advanced**     |       32 GiB | 1                  | 2                | Desktop (48–64 GB)     |
| **Great Wall**   |      128 GiB | 1                  | 2                | Server (192+ GB)       |

**Basic** is heavy but doable on any modern smartphone.  **Advanced** requires
a serious desktop (48–64 GB system RAM) — the typical Bitcoin / CS enthusiast
can afford it.  **Great Wall** requires server-class hardware (192+ GB); beyond
this threshold, users should consider a custom setup.  All profiles use Argon2d
v0x13 with salt `b"greatwall"` and produce a 32-byte digest.

#### Why p=1 for all tiers

Argon2 supports `p > 1` for password-hashing scenarios where the defender has a
hard wall-clock budget (e.g. ~1 s per login attempt).  More parallel lanes let
the defender pack more total work into that fixed time window, raising the cost
per attacker guess.

Great Wall inverts this assumption: **the delay is the product, not a
constraint to minimise**.  If the defender has 4 cores and sets `p = 4`, they
finish 4× faster — but then the outer iteration count must be 4× higher to
fill the same target delay.  The net result is identical wall-clock time and
total work, except now the attacker's GPU or ASIC can exploit 4 independent
lanes in parallel.

With `p = 1` the entire computation is a single sequential chain of
memory-dependent operations.  This maximises two security metrics:

1. **OOM gate** (what % of weaker adversaries are excluded).  Total memory =
   `p × m`.  With `p = 1` the OOM threshold equals `m` itself — an attacker
   cannot reduce memory usage by serialising lanes one at a time.
2. **Attacker/defender compute ratio** (how much faster a stronger adversary
   can go).  An attacker cannot parallelise *within* a single lane because each
   step depends on the previous one.  The defender's commodity CPU is
   near-optimal for sequential memory-hard work, so the ratio stays close to 1.

The only tier-variable parameter is therefore `m` (memory): it controls which
devices are OOM-gated.  The total work is determined by the number of external
iterations `N`, which is chosen to meet the user's target delay.

#### Why t=2 (two internal passes)

With `t = 1`, Argon2 fills the memory array in a single sequential pass.  Once
block *i+1* is computed, block *i* is never read again — an attacker can
exploit this by discarding blocks and recomputing them on demand, trading time
for memory (the classic time-memory tradeoff).

With `t ≥ 2`, the second pass re-reads the entire array in a data-dependent
pattern (Argon2d).  Every block from pass 1 might be referenced at any point
during pass 2, so the attacker must keep the full `m` allocation resident at
the pass boundary or face a quadratic (or worse) recomputation penalty.

`t = 2` is the minimum needed to guarantee that the full memory array must be
simultaneously resident.  `t > 2` would add marginal hardness at the cost of
proportionally more wall-clock time per external iteration — since `N` is
adjusted to fill the target delay anyway, `t = 2` is the sweet spot.

#### Why Argon2d (not Argon2i or Argon2id)

Argon2i uses data-independent memory access patterns to resist side-channel
attacks — critical when the secret is a short password typed into a shared
server.  Argon2id is a hybrid (Argon2i for the first pass, Argon2d thereafter).

In Great Wall the "secret" is 64 bits of entropy derived from a BIP-39
mnemonic, processed on the user's own device.  Side-channel resistance is
irrelevant: the user controls the hardware, and 64-bit entropy cannot be
brute-forced even with full knowledge of access patterns.

Argon2d's data-dependent addressing makes its memory-hardness strictly
stronger: the access pattern depends on the data, so an attacker cannot
precompute which memory cells will be needed and must keep the full allocation
resident.  This maximises the memory-hardness guarantee — exactly what we want
for the OOM gate.

#### Factoring external iterations

The user's target delay (hours, days, or weeks) determines the total number of
external iterations `N`.  To make the protocol more forgiving when the user
forgets the exact iteration count, `N` is factored into two levels:

```
N = n_checkpoints × iters_per_checkpoint
```

- **`iters_per_checkpoint`**: chosen so that one checkpoint interval takes a
  convenient wall-clock duration (e.g. 30 minutes or 1 hour).  The checkpoint
  file records every intermediate digest at checkpoint boundaries.
- **`n_checkpoints`**: the number of such intervals needed to reach the target
  delay.

If the user forgets whether they chose 48 or 50 hours, they only need to try a
few nearby checkpoint counts — each trial is a single Argon2d pass plus a
Stage 2 encode, not a full re-computation.  The checkpoint file also enables
resumption after interruption.

The user controls only the number of *external iterations* (hash-then-feed-back
cycles).  This determines the wall-clock time cost.

```
P1‖P2 (64 bits of raw entropy)
    │
    ▼
Argon2d(input, salt="greatwall", profile=Basic|Advanced|GreatWall)
    │   Iterated N times (configurable in the panel).
    │   Each iteration feeds the previous 32-byte digest back as input.
    │   Intermediate digests are checkpointed to disk for resume/trial-and-error.
    ▼
SHA-256(digest)[0:8]         First 8 bytes → uint64 *p*
    │                        Bits 0–31  → Re(p):  bit j ⟹ −2^{−(3+j)}
    │                        Bits 32–63 → Im(p):  bit j ⟹ −2^{−(3+j)}
    ▼
Stage 2 fractal with perturbation *p* is now active.
```

#### Checkpointing

The checkpoint file (`.argon2_checkpoint_{input_hex}_{profile}.bin`) stores a
sequence of 36-byte records: 4-byte LE iteration index + 32-byte digest.  This
serves two purposes:

1. **Resume after interruption**: on restart, the highest saved iteration ≤ the
   target is used as the starting point, avoiding redundant computation.
2. **Trial-and-error on iteration count**: if the user forgets the exact number
   of external iterations, they can try nearby values cheaply — each is a simple
   lookup + stage-2 encode, no re-hashing required.

### Stage 2: Encoding (mnemonic → P3, P4)

```
entropy[64:128]              Last 64 entropy bits → 2 chunks of 32 bits each.
    │
    ▼  (for each chunk)
encode(chunk, area, params,  Rust FFI: bisection tree using the PERTURBED
       p=p)                  fractal engine (escape_count_generic with *p*).
    ▼
P3, P4 displayed as crosshair markers on the Stage 2 viewport.
```

### Decoding (select mode: 2 clicked points → 64 entropy bits, per stage)

Triggered by pressing **S** to enter select mode, then clicking 2 points on
the fractal viewport.  Select mode is **stage-aware**: in stage 1 it selects
S1/S2, in stage 2 it selects S3/S4.

For each click:
1. Screen pixel → f64 complex coordinate → raw i64 Fixed (the only lossy step).
2. `decode_full()` validates the point (checks it stays inside every
   contracted child) and returns 32 bits + leaf rect + validity flag.
3. The path prefix is advanced after each valid point (re-encode the decoded
   bits to derive the path).

After 2 stage-1 points: 64 bits stored for Argon2.
After 2 stage-2 points: stage-1 + stage-2 bits combined → `bits_to_mnemonic()`
→ full 12-word BIP39 phrase.

### Manual bit input mode

Press **M** to enter manual mode.  The user types bits directly:

- **O** key = bit 0 (left for vertical splits, up for horizontal)
- **I** key = bit 1 (right for vertical splits, down for horizontal)
- **Backspace** = undo last bit (crosses point boundary if needed)

Manual mode enforces **2 × 32 bits** per stage.  After 32 bits, point 1 is
auto-committed (path prefix advances), and input continues for point 2.
After 64 total bits, the stage completes.  Area visualization auto-enables and
focuses on the latest bisection step.

The status bar shows `P1 bit N/32` or `P2 bit N/32` with the live path string.

### BIP39 phrase display and clipboard

The BIP39 text input field and Encode button are **only visible in debug mode**
(press **D** to toggle).  In normal mode, the panel shows the decoded
mnemonic status and a **Copy** button that copies the phrase to the system
clipboard (via `pygame.scrap` or `xclip`/`xsel`/`wl-copy` fallback).

### Critical constraints for lossless round-trip

1. **Same area, params, and path prefix.** Both encode and decode must use
   identical `ENCODE_AREA`, `GUI_PARAMS`, and path prefix.  Any mismatch
   causes the bisection trees to diverge, producing wrong bits.

2. **Raw i64 coordinates.** The encode path returns `point_re_raw` /
   `point_im_raw` as raw `i64` Fixed values. These are the *only*
   representation that guarantees lossless decode.

3. **Click precision.** When decoding by clicking, the screen→float→i64
   conversion must land inside the encoded point's leaf rectangle.  At the
   viewer's default zoom this is typically satisfied for points encoded with
   32 bits (leaf rectangles are much larger than a pixel).

### Render cache

The viewer optionally maintains a FIFO hash table cache (`--cache-size N`)
that stores `escape_count` results keyed on raw `(i64, i64)` Fixed
coordinates.  The cache is used exclusively by the rendering path
(`bs_render_viewport`) and has no effect on the deterministic
encode/decode bijection.  When panning at the same zoom level, most pixels
map to the same Fixed coordinates as the previous frame, yielding near-total
cache reuse and significantly faster re-renders.

---

## PointGen: Candidate Point Generation

Candidate points for island discovery are generated by `PointGen`, a
deterministic PRNG seeded from the fractal parameters and bisection path:

```
path_hash = splitmix64(path, rng_seed)
seed_lo   = wymix(wymix(o ^ s0, p ^ s1), wymix(q ^ s2, path_hash ^ s3))
seed_hi   = wymix(wymix(o ^ s4, q ^ s5), wymix(p ^ s6, path_hash ^ s7))
```

Both seed halves depend on **all four inputs** (o, p, q, path_hash) via
independent mixing orders with 8 distinct salt constants (from frac(e) bits
0–511). This ensures every input bit pseudorandomly affects both the Re and Im
coordinates of generated points.

Point generation is stateless:

```
re_raw = wymix(seed_lo ^ derive_salt_re, index)
im_raw = wymix(seed_hi ^ derive_salt_im, index)
re = area.re_min + (re_raw % re_range)     // [re_min, re_max)
im = area.im_min + (im_raw % im_range)     // [im_min, im_max)
```

---

## Point Validation in Selection Mode

When a user clicks a point on the fractal in selection mode, `decode_full`
runs the full 32-level bisection and checks whether the point survives
contraction at every level:

1. Run bisection (same as encoding) to determine which half the point falls in.
2. If the chosen half is the larger one, apply contraction.
3. Check `chosen.contains(point)` — if false, mark `valid = false`.
4. Repeat for all 32 levels.

Points that fail validation (fall in a contracted-away dead zone) are rejected
with a status bar message. Selection does not advance until a valid point is
clicked. This prevents accepting points that would decode to different bits
than expected.

---

## Stage-2 Parameter Derivation (Hash Byte Attribution)

The Argon2 digest (32 bytes) is hashed with SHA-256, and the first 24 bytes
of the resulting hash are split into three 8-byte big-endian uint64 values
in **alphabetical order**:

```
h = SHA-256(argon2_digest)
o = h[0:8]       // orbit seed (z₀)
p = h[8:16]      // additive perturbation
q = h[16:24]     // linear perturbation (εz term)
```

Each parameter encodes a complex number with magnitude bits and sign bits:

| Param | Magnitude bits | Sign bits | Baseline | Min exponent |
|-------|---------------|-----------|----------|-------------|
| o     | 31 per component | bit 31 (Re), bit 63 (Im) | none | 2^(-3) |
| p     | 31 per component | bit 31 (Re), bit 63 (Im) | 2^(-3) = 1/8 | 2^(-4) |
| q     | 31 per component | bit 31 (Re), bit 63 (Im) | none | 2^(-5) |

---

## Compile-Time Logging

All diagnostic logging in the Rust engine uses the `log_verbose!` macro,
which compiles to nothing by default. To enable logging:

```bash
cargo build --release --features verbose
```

Without the feature flag, the compiler strips all logging code and the
variables used only for log formatting — zero runtime cost.
