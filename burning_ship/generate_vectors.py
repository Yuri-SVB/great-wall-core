#!/usr/bin/env python3
"""Generate the full Cartesian product of test vectors."""
import subprocess, sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(SCRIPT_DIR, "cli.py")
OUT_DIR = os.path.join(SCRIPT_DIR, "test_vectors", "v0.1.0")

MODES = {
    "m": {
        "zeros": "0000000000000000",
        "ones": "ffffffffffffffff",
        "abandon": None,  # use --bip39
        "vanity1": "951df7837eb13d5a",
        "vanity2": "951df78327313d56",
        "vanity3": "df936139cbdefbc1",
        "vanity4": "df936139cbdefbc1",
    },
    "d": {
        "zeros": "00000000000000000000000000000000",
        "ones": "ffffffffffffffffffffffffffffffff",
        "abandon": None,
        "vanity1": "951df7837eb13d5a019d1669a0280dbd",
        "vanity2": "951df78327313d565e861493d47ba558",
        "vanity3": "df936139cbdefbc1acb91913d85e4f61",
        "vanity4": "df936139cbdefbc1acbc0ac2947a4858",
    },
    "l": {
        "zeros": "0000000000000000000000000000000000000000000000000000000000000000",
        "ones": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "abandon": None,
        "vanity1": "951df7837eb13d5a019d1669a0280dbd0951df7837eb13d5a019d1669a0280db",
        "vanity2": "951df78327313d565e861493d47ba558d951df78327313d565e861493d47ba55",
        "vanity3": "df936139cbdefbc1acb91913d85e4f614df936139cbdefbc1acb91913d85e4f6",
        "vanity4": "df936139cbdefbc1acbc0ac2947a4858ddf936139cbdefbc1acbc0ac2947a485",
    },
}

BIP39 = {
    ("m", "abandon"): "abandon abandon abandon abandon abandon able",
    ("d", "abandon"): "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
    ("l", "abandon"): "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art",
}

ITERATIONS = [0, 1, 2, 3]

os.makedirs(OUT_DIR, exist_ok=True)

mode_names = {"m": "mini", "d": "default", "l": "large"}
total = 0
for mode, seeds in MODES.items():
    for seed_name, entropy_hex in seeds.items():
        for iters in ITERATIONS:
            fname = f"{mode_names[mode]}_{seed_name}_iter{iters}.json"
            path = os.path.join(OUT_DIR, fname)
            if os.path.exists(path):
                print(f"  SKIP  {fname} (exists)")
                total += 1
                continue
            if entropy_hex is not None:
                cmd = [sys.executable, CLI, "encode",
                       "--entropy", entropy_hex,
                       "--profile", "b", "--iterations", str(iters), "--mode", mode]
            else:
                bip39 = BIP39.get((mode, seed_name))
                if bip39 is None:
                    continue
                cmd = [sys.executable, CLI, "encode",
                       "--bip39", bip39,
                       "--profile", "b", "--iterations", str(iters), "--mode", mode]
            print(f"  GEN   {fname} ...", end="", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                print(f" FAIL: {result.stderr.strip()}")
                continue
            with open(path, "w") as f:
                f.write(result.stdout)
            print(" OK")
            total += 1

print(f"\nTotal vectors: {total}")
