"""
Great Wall chained protocol — one 32-bit point per stage.

The protocol encodes **exactly one 32-bit point per stage**, and each stage is
its own fractal:

    one stage  =  one fractal  =  one haystack
    one point  =  one needle

The first stage (index 0) is always the public, canonical Burning Ship
(o = p = q = 0) — it is not a secret haystack the attacker must *find*.  Every
later stage's fractal parameters (o, p, q) are derived by hashing **all
preceding points** through the memory-hard chain::

    θ_k = SHA-256( Argon2^N( bits of points 0 .. k-1 ) )  →  (o, p, q)

Because stage ``k+1``'s fractal cannot be derived until stage ``k``'s point is
fixed (the next θ depends on every prior point), the cost of descriptively
bypassing the memory-hard derivation compounds across the chain: ``D_θ`` scales
with the number of stages while a single point stays ≈ 32 bits.

``n_stages = entropy_bits / BITS_PER_POINT``; of those, the first is canonical
and the remaining ``n_stages − 1`` are secret, chain-derived haystacks.

This module is the single source of truth for the chained pipeline; the CLI and
GUI both drive their encode/decode through it.
"""

from burning_ship_engine import encode, decode_full
from constants import (
    BITS_PER_POINT, ENCODE_AREA, GUI_PARAMS,
    CANONICAL_O, CANONICAL_P, CANONICAL_Q,
    n_stages_for,
)
from argon2_pipeline import derive_stage_params


# Version of the *chained protocol* (this orchestration layer), independent of
# the Rust ENGINE_VERSION (the single-fractal encode/decode algorithm, which is
# unchanged).  This is the token that locks great-wall-core to its authoritative
# design doc: the design lives only in `great-wall-docs/great-wall-core/
# DESIGN.md` (the single source of truth) and declares this same version, so the
# code and the doc are verifiably in sync — bump both together when the
# protocol's encode/decode behaviour changes.
#
# Pre-1.0 (semver): the protocol is UNSTABLE and still evolving; anything may
# change.  Lineage / roadmap:
#   0.1.0  two-stage / multiple-points-per-stage prototype (retroactive label)
#   0.2.0  one 32-bit point per stage, chained fractals          <- current
#   0.3.0+ planned: new parameter families, etc.
#   1.0.0  first STABLE protocol — comprehensive frozen test vectors are
#          (re)built then; until then they are intentionally provisional and a
#          version guard in the test harness flags any mismatch as STALE so a
#          stale vector can never show a false pass.
PROTOCOL_VERSION = "0.2.0"


def get_protocol_version():
    """Return the chained-protocol version string (see PROTOCOL_VERSION)."""
    return PROTOCOL_VERSION


class StageResult:
    """One stage of a chained encode: its fractal parameters and encoded point.

    Attributes:
        index:   stage index (0 = canonical first fractal).
        o, p, q: uint64 fractal parameters for this stage's fractal.
        chunk:   the 32 entropy bits encoded by this stage's point.
        result:  the engine ``EncodeResult`` (point, path, steps, final_rect).
        digest:  the 32-byte Argon2 digest that produced (o, p, q), or ``None``
                 for the canonical first stage.
        params:  the 9-tuple ``(o, o_re, o_im, p, p_re, p_im, q, q_re, q_im)``
                 with float display values, or ``None`` for stage 0.
    """

    def __init__(self, index, o, p, q, chunk, result, digest=None, params=None):
        self.index = index
        self.o = o
        self.p = p
        self.q = q
        self.chunk = chunk
        self.result = result
        self.digest = digest
        self.params = params

    @property
    def canonical(self):
        return self.index == 0

    @property
    def point(self):
        """The encoded point as ``(re, im, re_raw, im_raw)``."""
        r = self.result
        return (r.point_re, r.point_im, r.point_re_raw, r.point_im_raw)


def stage_params(index, prior_bits, profile, iterations,
                 progress_cb=None, stop_check=None):
    """Resolve (o, p, q, digest, params) for the stage at ``index``.

    Stage 0 is canonical (o = p = q = 0, no derivation).  Every later stage is
    derived from the cumulative bits of all preceding points via the
    memory-hard chain.
    """
    if index == 0:
        return CANONICAL_O, CANONICAL_P, CANONICAL_Q, None, None
    digest, params = derive_stage_params(
        prior_bits, profile, iterations, progress_cb, stop_check)
    o, _o_re, _o_im, p, _p_re, _p_im, q, _q_re, _q_im = params
    return o, p, q, digest, params


def encode_entropy(entropy_bits, profile, iterations,
                   progress_cb=None, stop_check=None):
    """Encode entropy bits into one chained point per stage.

    ``len(entropy_bits)`` must be a multiple of ``BITS_PER_POINT``.  Returns a
    list of :class:`StageResult`, one per stage, in order.

    ``progress_cb(stage_index, done_iters)`` (optional) reports Argon2 progress
    for each derived stage; ``stop_check()`` (optional) may raise to abort.
    """
    if len(entropy_bits) % BITS_PER_POINT != 0:
        raise ValueError(
            f"entropy length {len(entropy_bits)} is not a multiple of "
            f"{BITS_PER_POINT}")
    n = len(entropy_bits) // BITS_PER_POINT
    stages = []
    prior_bits = []
    for s in range(n):
        cb = (lambda done, _s=s: progress_cb(_s, done)) if progress_cb else None
        o, p, q, digest, params = stage_params(
            s, prior_bits, profile, iterations, cb, stop_check)
        chunk = entropy_bits[s * BITS_PER_POINT:(s + 1) * BITS_PER_POINT]
        result = encode(chunk, area=ENCODE_AREA, params=GUI_PARAMS,
                        o=o, p=p, q=q, path_prefix="O")
        stages.append(StageResult(s, o, p, q, chunk, result, digest, params))
        prior_bits = prior_bits + chunk
    return stages


def decode_entropy(points_raw, profile, iterations,
                   progress_cb=None, stop_check=None):
    """Decode chained points (one per stage) back into entropy bits.

    ``points_raw`` is a list of ``(re_raw, im_raw)`` i64 Fixed pairs, one per
    stage, in order.  Decoding mirrors encoding: each stage's fractal is
    re-derived from the bits decoded so far, so the chain must be walked in
    order.  Returns the flat list of decoded entropy bits.
    """
    out_bits = []
    prior_bits = []
    for s, (re_raw, im_raw) in enumerate(points_raw):
        cb = (lambda done, _s=s: progress_cb(_s, done)) if progress_cb else None
        o, p, q, _digest, _params = stage_params(
            s, prior_bits, profile, iterations, cb, stop_check)
        bits, _leaf, _valid, _path = decode_full(
            re_raw, im_raw, BITS_PER_POINT, area=ENCODE_AREA, params=GUI_PARAMS,
            o=o, p=p, q=q, path_prefix="O")
        out_bits.extend(bits)
        prior_bits = prior_bits + bits
    return out_bits


__all__ = [
    "StageResult", "stage_params", "encode_entropy", "decode_entropy",
    "n_stages_for", "PROTOCOL_VERSION", "get_protocol_version",
]
