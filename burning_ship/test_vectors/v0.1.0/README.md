# Test Vectors v0.1.0

Engine version: `0.1.0`

> **⚠️ PROVISIONAL — pre-1.0 protocol.** The chained protocol is still
> evolving (`protocol_version` `0.2.0`; more breaking changes are planned —
> new parameter families, etc.), so **comprehensive frozen vectors are
> deliberately deferred to the stable `1.0.0` release** rather than rebuilt for
> every interim bump. These vectors are stamped with their `protocol_version`,
> and `test_vectors.py` carries a **version guard**: any vector whose
> `protocol_version` differs from the current `protocol.PROTOCOL_VERSION` is
> reported **STALE** and skipped — never counted as a pass. That is what keeps
> deferral safe: a stale vector can never show false-green. The deterministic
> core round-trip and the chained derivation are verified separately (see
> `protocol.py` / commit history).

All vectors use Argon2 profile `b` (basic, 1 GiB, t=2, p=1).

These vectors target the **chained one-point-per-stage protocol**: one 32-bit
point per stage, `n_stages = entropy_bits / 32` (mini=2, default=4, large=8).
Stage 0 is canonical; every later stage's fractal is derived by hashing all
preceding points (Argon2^iters → SHA-256 → (o,p,q)). Each vector is a `stages[]`
document (`stages[i].params`, `stages[i].leaf`, `stages[i].argon2`).

## Generation policy

Because each Argon2 iteration now runs once per *secret* stage
(`n_stages − 1` chained derivations), `iters > 0` vectors are expensive for the
larger presets. The frozen set therefore covers the full breadth at **iter0**
(an instant identity derivation that still exercises the chain — every secret
stage gets distinct params from the growing prior-point prefix), plus a few
cheap real-Argon2 vectors so the meta tests have `iter1`/`iter2` inputs.

| Group | Files | Purpose |
|-------|-------|---------|
| iter0, all presets/seeds | `{mini,default,large}_{zeros,ones,abandon,vanity1,vanity2,vanity3,vanity4}_iter0.json` | Full-breadth frozen + round-trip coverage |
| real Argon2 (mini) | `mini_zeros_iter1.json`, `mini_zeros_iter2.json`, `mini_vanity1_iter1.json` | 1–2 actual Argon2 passes |
| real Argon2 (default) | `default_zeros_iter1.json` | A 4-stage secret with real derivation; feeds the meta tests |

### `zeros` / `ones` as a contraction stress test

The all-`0` (and all-`1`) vectors are not just boundary inputs — they are a
deliberate **stress test of the contraction heuristic**. An all-`0`/all-`1`
bisection path keeps selecting the same child, steering toward the sparse edge
"void" away from the island structure, where the chosen child is repeatedly the
*larger* one — so contraction `f(r) = (1 + 3r)/4` fires on nearly every level,
far more than on a mixed path. Because `f` is floored at `1/4`, the leaf shrinks
geometrically toward the void but stays **strictly positive** (these leaves are
~`2⁻²²` wide — small, never zero). Their clean round-trip confirms the
encoder still lands a valid positive-area leaf and decode's dead-zone validation
still holds under maximal contraction.

**Navigability preserved at all scales (the key positive result).** Smaller leaf
area does *not* mean a lost, structureless location. Encoding the canonical
stage-0 all-`0` and all-`1` points, **every one of the 32 bisection levels finds
good islands** — `≥ 32` (the `target_good` floor; min 32, max ~56–62) at every
level, with **zero** geometric-fallback steps (no level enters an island-less
void). So the path down to these contracted corner leaves is densely populated
with identifiable reference points at *every zoom scale*: the extreme corner
case remains navigable/recognizable, which is the a-priori evidence that the
small leaf is still surrounded by usable structure.

Regenerate with `python3 generate_vectors.py` (skips files that already exist).

## Cross-mode invariant

The first stage's leaf (stage 0) is identical across `mini_abandon_iter0`,
`default_abandon_iter0`, and `large_abandon_iter0`: all three share the same
first 32 bits of entropy (all-zero "abandon" words) on the same canonical
fractal, and the area tree is an invariant property of the fractal.

## Running

```bash
cd burning_ship
python3 test_vectors.py              # all tests
python3 test_vectors.py --verbose    # show diff details on failure
```
