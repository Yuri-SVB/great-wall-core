"""
Great Wall chained protocol — stage-0 text + one 32-bit point per later stage.

The protocol begins with a mandatory **stage 0 that carries no point — only a
short text input** — and then encodes **exactly one 32-bit point per later
stage**.  Every point-bearing stage is its own fractal::

    stage 0    =  text only (no point)  =  the salt/pepper that seeds the chain
    one stage  =  one fractal           =  one haystack   (point stages 1 .. N)
    one point  =  one needle

``N = entropy_bits / BITS_PER_POINT`` is the number of **point** stages; the
total stage count is ``N + 1`` because stage 0 is always present.  Every point
stage's fractal parameters (o, p, q) are derived from the memory-hard chain run
over **stage-0 text followed by the bits of all preceding points**::

    θ_k = SHA-256( Argon2^N( stage-0 text ‖ bits of points 1 .. k-1 ) )  →  (o, p, q)

Because stage 1's fractal already derives from stage-0 text, **there is no
longer a public "canonical" first fractal** — every point-bearing fractal is
private and chain-derived.  Because stage ``k+1``'s fractal cannot be derived
until stage ``k``'s point is fixed (the next θ depends on every prior point
*and* on stage-0 text), the cost of descriptively bypassing the memory-hard
derivation compounds across the chain: ``D_θ`` scales with the number of stages
while a single point stays ≈ 32 bits.

Internally the ``N`` point stages are addressed with 0-based indices
``0 .. N-1`` (point stage with index ``i`` is "stage ``i+1``" in the documented
1-based numbering); stage 0 (the text) is not a :class:`StageResult`.

This module is the single source of truth for the chained pipeline; the CLI and
GUI both drive their encode/decode through it.
"""

import struct

from burning_ship_engine import encode, decode_full, argon2id_master
from constants import (
    BITS_PER_POINT, ENCODE_AREA, GUI_PARAMS,
    ARGON2ID_MASTER_OUTPUT_BYTES, MASTER_DISPLAY_CHARS,
    n_stages_for,
)
from argon2_pipeline import derive_stage_params
from encoding import stage_text_bytes


# Version of the *chained protocol* (this orchestration layer), independent of
# the Rust ENGINE_VERSION (the single-fractal encode/decode algorithm, which is
# unchanged at 0.1.0).  This is the token that locks great-wall-core to its
# authoritative design doc: the design lives only in `great-wall-docs/
# great-wall-core/DESIGN.md` (the single source of truth) and declares this same
# version, so the code and the doc are verifiably in sync — bump both together
# when the protocol's encode/decode behaviour changes.
#
# Pre-1.0 (semver): the protocol is UNSTABLE and still evolving; anything may
# change.  Lineage / roadmap:
#   0.1.0  two-stage / multiple-points-per-stage prototype (retroactive label)
#   0.2.0  one 32-bit point per stage, canonical first fractal, SHA512 carry-over
#   0.3.0  stage-0 text + Argon2id carry-over (one point per LATER stage)  <- current
#   1.0.0  first STABLE protocol — comprehensive frozen test vectors are
#          (re)built then; until then they are intentionally provisional and a
#          version guard in the test harness flags any mismatch as STALE so a
#          stale vector can never show a false pass.
PROTOCOL_VERSION = "0.3.0"


def get_protocol_version():
    """Return the chained-protocol version string (see PROTOCOL_VERSION)."""
    return PROTOCOL_VERSION


class StageResult:
    """One point stage of a chained encode: its fractal params and encoded point.

    Attributes:
        index:   0-based point-stage index (documented "stage index+1").
        o, p, q: uint64 fractal parameters for this stage's fractal.
        chunk:   the 32 entropy bits encoded by this stage's point.
        result:  the engine ``EncodeResult`` (point, path, steps, final_rect).
        digest:  the 32-byte Argon2 digest that produced (o, p, q).
        params:  the 9-tuple ``(o, o_re, o_im, p, p_re, p_im, q, q_re, q_im)``
                 with float display values.

    Under protocol 0.3.0 every point stage is chain-derived (there is no
    canonical stage), so ``digest`` and ``params`` are always set.
    """

    def __init__(self, index, o, p, q, chunk, result, digest, params):
        self.index = index
        self.o = o
        self.p = p
        self.q = q
        self.chunk = chunk
        self.result = result
        self.digest = digest
        self.params = params

    @property
    def point(self):
        """The encoded point as ``(re, im, re_raw, im_raw)``."""
        r = self.result
        return (r.point_re, r.point_im, r.point_re_raw, r.point_im_raw)

    @property
    def leaf_center_raw(self):
        """The raw (re, im) i64 centre of this stage's encoded-point leaf rect."""
        return _leaf_center_raw(self.result.final_rect)


def _midpoint(a, b):
    """Replicate Rust's Fixed::midpoint: (a>>1) + (b>>1) + (a & b & 1)."""
    return (a >> 1) + (b >> 1) + (a & b & 1)


def _leaf_center_raw(rect):
    """Raw (re, im) i64 centre of a leaf rectangle (I4F60 fixed-point)."""
    return _midpoint(rect.re_min, rect.re_max), _midpoint(rect.im_min, rect.im_max)


def stage_params(index, stage0_text, prior_bits, profile, iterations,
                 progress_cb=None, stop_check=None):
    """Resolve (o, p, q, digest, params) for the point stage at ``index``.

    Every point stage (``index`` ≥ 0, including the first) is chain-derived from
    stage-0 text plus the cumulative bits of all preceding points via the
    memory-hard chain — there is no canonical stage in protocol 0.3.0.
    """
    digest, params = derive_stage_params(
        stage0_text, prior_bits, profile, iterations, progress_cb, stop_check)
    o, _o_re, _o_im, p, _p_re, _p_im, q, _q_re, _q_im = params
    return o, p, q, digest, params


def encode_entropy(entropy_bits, stage0_text, profile, iterations,
                   progress_cb=None, stop_check=None):
    """Encode entropy bits into one chained point per (later) stage.

    ``len(entropy_bits)`` must be a multiple of ``BITS_PER_POINT``.  ``stage0_text``
    is the mandatory stage-0 text input that seeds the chain (normalized to
    ``[A-Z0-9-]``; may be empty).  Returns a list of :class:`StageResult`, one
    per point stage, in order.

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
            s, stage0_text, prior_bits, profile, iterations, cb, stop_check)
        chunk = entropy_bits[s * BITS_PER_POINT:(s + 1) * BITS_PER_POINT]
        result = encode(chunk, area=ENCODE_AREA, params=GUI_PARAMS,
                        o=o, p=p, q=q, path_prefix="O")
        stages.append(StageResult(s, o, p, q, chunk, result, digest, params))
        prior_bits = prior_bits + chunk
    return stages


def decode_entropy(points_raw, stage0_text, profile, iterations,
                   progress_cb=None, stop_check=None):
    """Decode chained points (one per stage) back into entropy bits.

    ``points_raw`` is a list of ``(re_raw, im_raw)`` i64 Fixed pairs, one per
    point stage, in order.  ``stage0_text`` is the stage-0 text that seeded the
    chain.  Decoding mirrors encoding: each stage's fractal is re-derived from
    stage-0 text plus the bits decoded so far, so the chain must be walked in
    order.  Returns the flat list of decoded entropy bits.
    """
    out_bits = []
    prior_bits = []
    for s, (re_raw, im_raw) in enumerate(points_raw):
        cb = (lambda done, _s=s: progress_cb(_s, done)) if progress_cb else None
        o, p, q, _digest, _params = stage_params(
            s, stage0_text, prior_bits, profile, iterations, cb, stop_check)
        bits, _leaf, _valid, _path = decode_full(
            re_raw, im_raw, BITS_PER_POINT, area=ENCODE_AREA, params=GUI_PARAMS,
            o=o, p=p, q=q, path_prefix="O")
        out_bits.extend(bits)
        prior_bits = prior_bits + bits
    return out_bits


# ---------------------------------------------------------------------------
# Master-secret export (final Argon2id over the setup transcript)
# ---------------------------------------------------------------------------

def build_export_transcript(stage0_text, iterations, stage_records, export_label):
    """Serialize the reproducible setup transcript for the master-secret export.

    Mirrors DESIGN.md "Master-Secret Export": the message is the stage-0 input,
    the iteration count, and — for every point stage up to and including the
    exporting stage ``k`` — its derived params ``(o, p, q)`` and the centre of
    its encoded point's leaf rectangle, with the exporting stage's own text
    label appended.

    ``stage_records`` is an ordered list (stages 0..k) of
    ``(o, p, q, leaf_re_raw, leaf_im_raw)`` tuples.

    Byte layout (this module defines the exact serialization; the DESIGN
    constrains only the field order and contents):

        u16_be len ‖ stage-0 text bytes (ASCII [A-Z0-9-])
        u32_be iterations
        for each stage record:
            u64_be o ‖ u64_be p ‖ u64_be q
            i64_be leaf_centre_re ‖ i64_be leaf_centre_im
        u16_be len ‖ export-label bytes (ASCII [A-Z0-9-])

    Variable-length texts are length-prefixed so the message parses
    unambiguously and reproduces bit-for-bit on recovery.
    """
    def _u64(x):
        return struct.pack(">Q", x & 0xFFFFFFFFFFFFFFFF)

    def _i64(x):
        return struct.pack(">q", x)

    def _text(t):
        b = stage_text_bytes(t)
        return struct.pack(">H", len(b)) + b

    msg = bytearray()
    msg += _text(stage0_text)
    msg += struct.pack(">I", iterations & 0xFFFFFFFF)
    for (o, p, q, leaf_re, leaf_im) in stage_records:
        msg += _u64(o) + _u64(p) + _u64(q)
        msg += _i64(leaf_re) + _i64(leaf_im)
    msg += _text(export_label)
    return bytes(msg)


def export_master_secret(stage0_text, iterations, stage_records, export_label,
                         out_len=ARGON2ID_MASTER_OUTPUT_BYTES):
    """Run the master-secret export (one Argon2id pass over the transcript).

    Returns the raw ``out_len``-byte Argon2id output.  ``stage_records`` covers
    stages 0..k (the exporting stage ``k`` inclusive); see
    :func:`build_export_transcript`.  Use :func:`master_secret_display` for the
    conventional 32-character default view.
    """
    message = build_export_transcript(
        stage0_text, iterations, stage_records, export_label)
    return argon2id_master(message, out_len)


def master_secret_display(raw):
    """Conventional default view of a master secret: the first 32 hex chars.

    DESIGN.md gates the full 1024-byte output behind advanced options and shows
    a conventional 32 characters by default; that gating is deferred, so for the
    time being the export surfaces only the first ``MASTER_DISPLAY_CHARS`` hex
    characters of the Argon2id output.
    """
    return raw.hex()[:MASTER_DISPLAY_CHARS]


def export_master_secret_from_stages(stage0_text, iterations, stages,
                                     export_label, export_stage_index=None,
                                     out_len=ARGON2ID_MASTER_OUTPUT_BYTES):
    """Convenience wrapper: build the transcript from :class:`StageResult` list.

    ``stages`` is the ordered point-stage list from :func:`encode_entropy`.
    ``export_stage_index`` is the 0-based point stage to export at (default: the
    last stage); stages 0..export_stage_index are included in the transcript.
    """
    if export_stage_index is None:
        export_stage_index = len(stages) - 1
    records = []
    for sr in stages[:export_stage_index + 1]:
        leaf_re, leaf_im = sr.leaf_center_raw
        records.append((sr.o, sr.p, sr.q, leaf_re, leaf_im))
    return export_master_secret(
        stage0_text, iterations, records, export_label, out_len)


__all__ = [
    "StageResult", "stage_params", "encode_entropy", "decode_entropy",
    "build_export_transcript", "export_master_secret",
    "export_master_secret_from_stages", "master_secret_display",
    "n_stages_for", "PROTOCOL_VERSION", "get_protocol_version",
]
