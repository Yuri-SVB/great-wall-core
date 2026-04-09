#!/usr/bin/env bash
#
# Run all Great Wall tests in one go.
#
# Usage:
#   cd burning_ship
#   bash run_tests.sh
#
# Exit code: 0 if all pass, 1 if any fail.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

FAIL=0

echo "============================================"
echo "  Rust unit tests"
echo "============================================"
if (cd rust_engine && cargo test 2>&1); then
    echo "  => Rust tests OK"
else
    echo "  => Rust tests FAILED"
    FAIL=1
fi

echo ""
echo "============================================"
echo "  Bijection test (1-8 bits)"
echo "============================================"
if python3 test_bijection.py; then
    echo "  => Bijection test OK"
else
    echo "  => Bijection test FAILED"
    FAIL=1
fi

echo ""
echo "============================================"
echo "  Frozen vectors + round-trips + meta tests"
echo "============================================"
if python3 test_vectors.py; then
    echo "  => Vector tests OK"
else
    echo "  => Vector tests FAILED"
    FAIL=1
fi

echo ""
echo "============================================"
if [ $FAIL -eq 0 ]; then
    echo "  ALL TESTS PASSED"
else
    echo "  SOME TESTS FAILED"
fi
echo "============================================"

exit $FAIL
