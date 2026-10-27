"""Check the engine against the frozen landmark-path vectors.

Every vector is recomputed and compared exactly: stops, cell bitmaps (by
digest) and hops. The vectors live under the engine's landmark-path version;
an output change without a version bump fails here.
"""
import glob
import json
import os
import sys

import burning_ship_engine as eng
from generate_landmark_vectors import stop_doc

VECTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_vectors",
                          f"landmark-v{eng.landmark_path_version()}")
INVALID_PARAMS = 1


def check(path):
    with open(path) as f:
        doc = json.load(f)
    assert doc["version"] == eng.landmark_path_version(), "version mismatch"
    b, leaf = doc["board"], doc["leaf"]
    status, stops, hops = eng.landmark_path(
        eng.Rect(*leaf["rect"]), leaf["path"], doc["params"], b["o"], b["p"], b["q"])
    assert status == 0, f"status {status}"
    got = [stop_doc(s) for s in stops]
    assert len(got) == len(doc["stops"]), f"{len(got)} stops, expected {len(doc['stops'])}"
    for k, (g, want) in enumerate(zip(got, doc["stops"])):
        assert g == want, f"stop {k}: {g} != {want}"
    assert hops == doc["hops"], f"hops {hops} != {doc['hops']}"


def check_invalid_params():
    params = dict(eng.landmark_params(), rho_den=0)
    status, stops, hops = eng.landmark_path(eng.Rect(0, 1, 0, 1), "O", params)
    assert (status, stops, hops) == (INVALID_PARAMS, [], []), (status, stops, hops)


def main():
    files = sorted(glob.glob(os.path.join(VECTOR_DIR, "landmark_*.json")))
    if not files:
        print(f"no vectors in {VECTOR_DIR}: a landmark-path version bump needs new vectors")
        return 1
    failed = 0
    for path in files:
        try:
            check(path)
            print(f"  ok   {os.path.basename(path)}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {os.path.basename(path)}: {e}")
    check_invalid_params()
    print("  ok   invalid parameters refused")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
