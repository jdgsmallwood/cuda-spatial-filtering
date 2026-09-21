#!/usr/bin/env python3
"""
Simulate the RedisBeamFFTWriter key-value mapping logic to prove
(or disprove) channel cross-pollination.

Replicates the C++ logic from writers.hpp:
  - get_key_index(ch, pol, beam, f)
  - precomputed_keys construction
  - slot_madd_args_ pre-seeding
  - process_block value assignment

Uses synthetic FFT data where fft_output[ch][pol][beam][f] = float(ch),
so any cross-pollination would be immediately visible as wrong values.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ---- Parameters (adjust to match a real build) ----
NR_CHANNELS       = 4
NR_POLARIZATIONS  = 2
NR_BEAMS          = 2
NR_FREQS          = 8       # FFT_SIZE / DOWNSAMPLE_FACTOR
CHANNELS_PER_WRITE = 2      # round-robin: 2 channels per call -> num_slots = 2


# ---- Replicate get_key_index from writers.hpp line 1127 ----
def get_key_index(ch, pol, beam, f):
    return (ch * (NR_POLARIZATIONS * NR_FREQS * NR_BEAMS)
            + pol * (NR_FREQS * NR_BEAMS)
            + beam * NR_FREQS
            + f)


# ---- Replicate precomputed_keys construction (lines 1029-1058) ----
precomputed_keys: List[str] = [""] * (NR_CHANNELS * NR_POLARIZATIONS * NR_FREQS * NR_BEAMS)
precomputed_max_keys: List[str] = [""] * len(precomputed_keys)

for ch in range(NR_CHANNELS):
    for pol in range(NR_POLARIZATIONS):
        for beam in range(NR_BEAMS):
            for f in range(NR_FREQS):
                f_shifted = (f + NR_FREQS // 2) % NR_FREQS
                key = f"ts:fft:ch:{ch}:p:{pol}:b:{beam}:f:{f_shifted}"
                max_key = f"ts:fft_max1s:ch:{ch}:p:{pol}:b:{beam}:f:{f_shifted}"
                precomputed_keys[get_key_index(ch, pol, beam, f)] = key
                precomputed_max_keys[get_key_index(ch, pol, beam, f)] = max_key


# ---- Replicate slot_madd_args_ pre-seeding (lines 1069-1089) ----
channels_per_write = CHANNELS_PER_WRITE if CHANNELS_PER_WRITE > 0 else NR_CHANNELS
num_slots = (NR_CHANNELS + channels_per_write - 1) // channels_per_write

slot_keys: List[List[str]] = []  # slot_keys[slot][data_point_idx] = redis_key

for s in range(num_slots):
    ch_start = s * channels_per_write
    ch_end = min(ch_start + channels_per_write, NR_CHANNELS)
    channels_in_slot = ch_end - ch_start

    slot_size = channels_in_slot * NR_POLARIZATIONS * NR_BEAMS * NR_FREQS
    keys_for_slot: List[str] = [""] * slot_size

    idx = 0
    for ch in range(ch_start, ch_end):
        for pol in range(NR_POLARIZATIONS):
            for beam in range(NR_BEAMS):
                for f in range(NR_FREQS):
                    keys_for_slot[idx] = precomputed_keys[get_key_index(ch, pol, beam, f)]
                    idx += 1

    slot_keys.append(keys_for_slot)


# ---- Synthetic FFT block: fft_output[ch][pol][beam][f] = float(ch) ----
# This means channel N contributes a constant value of N everywhere.
# Any cross-pollination would produce a key for channel A with value B (A != B).
fft_output = np.zeros((NR_CHANNELS, NR_POLARIZATIONS, NR_BEAMS, NR_FREQS), dtype=np.float32)
for ch in range(NR_CHANNELS):
    fft_output[ch, :, :, :] = float(ch)


# ---- Simulate process_block for each slot (lines 1092-1122) ----
print("=" * 70)
print(f"Config: NR_CHANNELS={NR_CHANNELS}, NR_POL={NR_POLARIZATIONS}, "
      f"NR_BEAMS={NR_BEAMS}, NR_FREQS={NR_FREQS}, channels_per_write={channels_per_write}")
print(f"Synthetic data: fft_output[ch][pol][beam][f] = float(ch)")
print("=" * 70)

errors: List[str] = []
all_assignments: Dict[str, float] = {}  # key -> value seen in Redis TS.MADD

for current_slot in range(num_slots):
    ch_start = current_slot * channels_per_write
    ch_end = min(ch_start + channels_per_write, NR_CHANNELS)

    current_idx = 0
    for ch in range(ch_start, ch_end):
        for pol in range(NR_POLARIZATIONS):
            for beam in range(NR_BEAMS):
                for f in range(NR_FREQS):
                    key = slot_keys[current_slot][current_idx]
                    cval = fft_output[ch, pol, beam, f]

                    # Parse the expected channel from the key name
                    # Key format: ts:fft:ch:{ch}:p:{pol}:b:{beam}:f:{f_shifted}
                    parts = key.split(":")
                    key_ch  = int(parts[3])
                    key_pol = int(parts[5])
                    key_beam = int(parts[7])

                    # Expected value for this key: float(key_ch)
                    expected_val = float(key_ch)

                    all_assignments[key] = cval

                    if abs(cval - expected_val) > 1e-6:
                        errors.append(
                            f"CROSS-POLLINATION DETECTED!\n"
                            f"  Slot={current_slot}, data (ch={ch},pol={pol},beam={beam},f={f})\n"
                            f"  Key={key} (belongs to ch={key_ch})\n"
                            f"  Value written={cval:.1f}, expected={expected_val:.1f}"
                        )

                    current_idx += 1

# ---- Report ----
print(f"\nTotal Redis TS.MADD assignments simulated: {len(all_assignments)}")
print(f"Expected total: {NR_CHANNELS} slots × "
      f"{channels_per_write * NR_POLARIZATIONS * NR_BEAMS * NR_FREQS} entries = "
      f"{NR_CHANNELS * NR_POLARIZATIONS * NR_BEAMS * NR_FREQS}")

if errors:
    print(f"\nFAIL: {len(errors)} cross-pollination errors found!")
    for e in errors:
        print(e)
else:
    print("\nPASS: No channel cross-pollination detected.")
    print("Every key for channel N received exactly value N.")

# ---- Also check: every key is written exactly once ----
key_counts: Dict[str, int] = {}
for current_slot in range(num_slots):
    ch_start = current_slot * channels_per_write
    ch_end = min(ch_start + channels_per_write, NR_CHANNELS)

    current_idx = 0
    for ch in range(ch_start, ch_end):
        for pol in range(NR_POLARIZATIONS):
            for beam in range(NR_BEAMS):
                for f in range(NR_FREQS):
                    key = slot_keys[current_slot][current_idx]
                    key_counts[key] = key_counts.get(key, 0) + 1
                    current_idx += 1

duplicates = {k: v for k, v in key_counts.items() if v != 1}
missing = set(precomputed_keys) - set(key_counts.keys())

if duplicates:
    print(f"\nWARNING: {len(duplicates)} keys appear in more than one slot!")
    for k, v in list(duplicates.items())[:5]:
        print(f"  {k}: {v} times")
else:
    print("PASS: Each Redis key appears in exactly one slot (no duplicates).")

if missing:
    print(f"\nWARNING: {len(missing)} keys never written (missing from all slots)!")
    for k in list(missing)[:5]:
        print(f"  {k}")
else:
    print("PASS: All precomputed keys are written exactly once across all slots.")

# ---- Show the full mapping for small configs ----
if NR_CHANNELS * NR_POLARIZATIONS * NR_BEAMS * NR_FREQS <= 64:
    print("\nFull key->value mapping:")
    for current_slot in range(num_slots):
        ch_start = current_slot * channels_per_write
        ch_end = min(ch_start + channels_per_write, NR_CHANNELS)
        print(f"\n  Slot {current_slot} (channels {ch_start}..{ch_end-1}):")
        idx = 0
        for ch in range(ch_start, ch_end):
            for pol in range(NR_POLARIZATIONS):
                for beam in range(NR_BEAMS):
                    for f in range(NR_FREQS):
                        key = slot_keys[current_slot][idx]
                        val = fft_output[ch, pol, beam, f]
                        print(f"    [{idx:2d}] key={key:<45s}  value={val:.1f}")
                        idx += 1
