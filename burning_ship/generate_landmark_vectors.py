"""Generate the frozen landmark-path vectors (test_vectors/landmark-v<version>/).

Each vector is a board (o, p, q) and a leaf encoded on it with the protocol's
discovery parameters, and the default landmark path of that leaf under the
engine's default landmark parameters: every stop's identity and measures, a
digest of its cell bitmap, and the hop distances. test_landmark_vectors.py
recomputes each path and compares it exactly.

Run:  python3 generate_landmark_vectors.py   (from burning_ship/; ~10 s a vector)
"""
import hashlib
import json
import os
import random

import burning_ship_engine as eng
from constants import BITS_PER_POINT, ENCODE_AREA, GUI_PARAMS

# (name, rng seed, perturbed board?)
VECTORS = (("plain", 1, False), ("perturbed", 2, True))
U64 = 1 << 64
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_vectors",
                       f"landmark-v{eng.landmark_path_version()}")


def stop_doc(stop):
    """A stop as frozen: its fields, with the bitmap as a SHA-256 digest."""
    doc = {k: v for k, v in stop.items() if k != "cells"}
    doc["cells_sha256"] = hashlib.sha256(stop["cells"]).hexdigest()
    return doc


def build(seed, perturbed):
    rng = random.Random(seed)
    o, p, q = (rng.randrange(U64) for _ in range(3)) if perturbed else (0, 0, 0)
    bits = [rng.randrange(2) for _ in range(BITS_PER_POINT)]
    leaf = eng.encode(bits, area=ENCODE_AREA, params=GUI_PARAMS, o=o, p=p, q=q)
    rect = leaf.final_rect
    params = eng.landmark_params()
    status, stops, hops = eng.landmark_path(rect, leaf.path, params, o, p, q)
    assert status == 0, f"landmark path status {status}"
    return {
        "version": eng.landmark_path_version(),
        "board": {"o": o, "p": p, "q": q},
        "bits": bits,
        "leaf": {"rect": [rect.re_min, rect.re_max, rect.im_min, rect.im_max],
                 "path": leaf.path},
        "params": params,
        "stops": [stop_doc(s) for s in stops],
        "hops": hops,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, seed, perturbed in VECTORS:
        doc = build(seed, perturbed)
        path = os.path.join(OUT_DIR, f"landmark_{name}.json")
        with open(path, "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        print(f"{path}: {len(doc['stops'])} stops")


if __name__ == "__main__":
    main()
