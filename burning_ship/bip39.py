"""
Minimal BIP39 mnemonic ↔ bit-array conversion.

Supports 6, 12, and 24 word mnemonics:
  6 words  →  66 bits ( 64 entropy +  2 checksum)
 12 words  → 132 bits (128 entropy +  4 checksum)
 24 words  → 264 bits (256 entropy +  8 checksum)
"""

import hashlib
import os

_dir = os.path.dirname(os.path.abspath(__file__))
_wordlist_path = os.path.join(_dir, "bip39_english.txt")

with open(_wordlist_path) as f:
    WORDLIST = [w.strip() for w in f if w.strip()]

# BIP39 protocol constants
BIP39_WORDLIST_SIZE = 2048
BITS_PER_WORD = 11            # log2(2048)
BITS_PER_BYTE = 8

# Supported configurations: word_count → (entropy_bits, checksum_bits)
BIP39_CONFIGS = {
    6:  (64,  2),
    12: (128, 4),
    24: (256, 8),
}

# Legacy aliases for 12-word default
BIP39_WORD_COUNT = 12
BIP39_ENTROPY_BITS = 128
BIP39_CHECKSUM_BITS = 4
BIP39_TOTAL_BITS = BIP39_ENTROPY_BITS + BIP39_CHECKSUM_BITS  # 132
BIP39_ENTROPY_CHUNKS = 4
BIP39_ENTROPY_CHUNK_SIZE = BIP39_ENTROPY_BITS // BIP39_ENTROPY_CHUNKS  # 32
BIP39_TOTAL_CHUNKS = 4
BIP39_TOTAL_CHUNK_SIZE = BIP39_TOTAL_BITS // BIP39_TOTAL_CHUNKS  # 33

assert len(WORDLIST) == BIP39_WORDLIST_SIZE, f"Expected {BIP39_WORDLIST_SIZE} words, got {len(WORDLIST)}"

_WORD_TO_INDEX = {w: i for i, w in enumerate(WORDLIST)}


def _checksum_bits(entropy_bits_list):
    """Compute BIP39 checksum bits for an entropy bit list.

    Checksum length = len(entropy) / 32.
    """
    n_entropy = len(entropy_bits_list)
    cs_len = n_entropy // 32

    entropy_bytes = bytearray()
    for i in range(0, n_entropy, BITS_PER_BYTE):
        byte_val = 0
        for j in range(BITS_PER_BYTE):
            byte_val = (byte_val << 1) | entropy_bits_list[i + j]
        entropy_bytes.append(byte_val)

    sha = hashlib.sha256(bytes(entropy_bytes)).digest()
    return [(sha[i // 8] >> (7 - i % 8)) & 1 for i in range(cs_len)]


def mnemonic_to_bits(mnemonic: str) -> list:
    """Convert a BIP39 mnemonic (6, 12, or 24 words) to bits.

    Returns the full bit list including checksum.
    Validates checksum. Raises ValueError on invalid input.
    """
    words = mnemonic.strip().lower().split()
    n = len(words)
    if n not in BIP39_CONFIGS:
        raise ValueError(f"Expected 6, 12, or 24 words, got {n}")

    entropy_len, cs_len = BIP39_CONFIGS[n]
    total_bits = entropy_len + cs_len

    for w in words:
        if w not in _WORD_TO_INDEX:
            raise ValueError(f"Unknown BIP39 word: '{w}'")

    bits = []
    for w in words:
        idx = _WORD_TO_INDEX[w]
        for bit_pos in range(BITS_PER_WORD - 1, -1, -1):
            bits.append((idx >> bit_pos) & 1)

    assert len(bits) == total_bits

    entropy = bits[:entropy_len]
    checksum = bits[entropy_len:]
    expected = _checksum_bits(entropy)

    if checksum != expected:
        raise ValueError("Invalid BIP39 checksum")

    return bits


def bits_to_mnemonic(bits: list) -> str:
    """Convert entropy bits to a BIP39 mnemonic.

    Accepts 64, 128, or 256 entropy bits (checksum is computed and appended),
    or the full bit length including checksum (66, 132, 264).
    """
    n = len(bits)

    # Check if this is a valid entropy length
    for wc, (ent, cs) in BIP39_CONFIGS.items():
        if n == ent:
            bits = bits + _checksum_bits(bits)
            n = len(bits)
            break
        if n == ent + cs:
            break
    else:
        valid = sorted(set(v for ent, cs in BIP39_CONFIGS.values() for v in (ent, ent + cs)))
        raise ValueError(f"Expected {valid} bits, got {n}")

    words = []
    for i in range(0, n, BITS_PER_WORD):
        idx = 0
        for j in range(BITS_PER_WORD):
            idx = (idx << 1) | bits[i + j]
        words.append(WORDLIST[idx])

    return " ".join(words)


def split_132_to_4x33(bits: list) -> list:
    """Split 132 bits into 4 chunks of 33 bits each."""
    assert len(bits) == BIP39_TOTAL_BITS
    return [bits[i*BIP39_TOTAL_CHUNK_SIZE:(i+1)*BIP39_TOTAL_CHUNK_SIZE] for i in range(BIP39_TOTAL_CHUNKS)]


def join_4x33_to_132(chunks: list) -> list:
    """Join 4 chunks of 33 bits into 132 bits."""
    assert len(chunks) == BIP39_TOTAL_CHUNKS
    assert all(len(c) == BIP39_TOTAL_CHUNK_SIZE for c in chunks)
    result = []
    for c in chunks:
        result.extend(c)
    return result


def split_128_to_4x32(bits: list) -> list:
    """Split 128 entropy bits into 4 chunks of 32 bits each."""
    assert len(bits) == BIP39_ENTROPY_BITS
    return [bits[i*BIP39_ENTROPY_CHUNK_SIZE:(i+1)*BIP39_ENTROPY_CHUNK_SIZE] for i in range(BIP39_ENTROPY_CHUNKS)]


def join_4x32_to_128(chunks: list) -> list:
    """Join 4 chunks of 32 bits into 128 bits."""
    assert len(chunks) == BIP39_ENTROPY_CHUNKS
    assert all(len(c) == BIP39_ENTROPY_CHUNK_SIZE for c in chunks)
    result = []
    for c in chunks:
        result.extend(c)
    return result
