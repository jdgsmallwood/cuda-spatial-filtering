#!/usr/bin/env python3
"""
Integration test: drives bin/observe_2_8 against a synthetic two-FPGA PCAP
with different stream-antenna-map JSONs and verifies that each map places
non-zero visibilities at exactly the canonical baselines predicted by the
stream→antenna assignment.

PCAP signal pattern (injected once, reused across all map cases):
  FPGA 0, recv 0 → int8 (4+0j)  for all 8 channels and 64 time steps
  FPGA 1, recv 0 → int8 (2+0j)  for all 8 channels and 64 time steps
  All other receivers → (0+0j)

This gives analytically predictable visibility values:
  autocorr FPGA-0 side  = NT × 4² = 262 144   (NT = 256 pkts × 64 steps = 16 384)
  autocorr FPGA-1 side  = NT × 2² =  65 536
  cross-correlation      = NT × 4 × 2 = 131 072

Three test cases vary the stream→antenna-ID assignment:
  A) FPGA0→ant10, FPGA1→ant20   canonical=[ant10, ant20], autocorr0=262144
  B) FPGA0→ant20, FPGA1→ant10   canonical=[ant10, ant20], autocorr0= 65536  ← swapped!
  C) FPGA0→ant5,  FPGA1→ant15   canonical=[ant5,  ant15], same structure as A

Each case asserts:
  1. antenna_ids[0:2] = [min_ant, max_ant] (canonical ascending-ID order)
  2. bl(0,0), bl(0,1), bl(1,1) are the only non-zero baselines in channel 0
  3. Autocorrelation values at bl(0,0)/bl(1,1) match which FPGA feeds canonical-0
  4. Cross-correlation bl(0,1) ≈ V_CROSS (same for both test cases A and B)
  5. baseline_antenna_ids and baseline_ids metadata are correct
  6. baseline_zeroed = 0 for the two active baselines, 1 everywhere else

Requirements:
  - GPU available (this is a CUDA pipeline; tests cannot run in a sandbox)
  - h5py and numpy installed
  - config.json, weights.json, alveo_delays.json, nr-signal-eigenvalues.json
    present in --workdir (workspace root by default)

Usage (from /workspace):
  python scripts/test_antenna_permutation.py
  python scripts/test_antenna_permutation.py --keep-artifacts --timeout 180
"""

import argparse
import json
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError as exc:
    print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
    print("  pip install h5py numpy", file=sys.stderr)
    sys.exit(1)

# ── Compile-time constants matching observe_2_8 ───────────────────────────────
NR_RECV_PER_PKT = 10  # NR_OBSERVING_RECEIVERS_PER_PACKET
NR_POL = 2
NR_TIME = 64  # NR_TIME_STEPS_PER_PACKET (hardcoded in observe.cu)
NR_CHAN = 8  # observe_2_8: NR_OBSERVING_CHANNELS = 8
NR_PKTS = 256  # NR_OBSERVING_PACKETS_FOR_CORRELATION
NR_FPGAS = 2  # observe_2_8: NR_OBSERVING_FPGA_SOURCES = 2
FPGA_IDS = [0, 1]
NR_RECEIVERS = NR_FPGAS * NR_RECV_PER_PKT  # = 20
NR_BASELINES = NR_RECEIVERS * (NR_RECEIVERS + 1) // 2  # = 210

# Signal on recv 0 of each FPGA; signed int8 (real, imag).
# Different magnitudes so autocorrs are distinguishable.
SIG = {0: (4, 0), 1: (2, 0)}

# Expected visibility values: NT = NR_PKTS × NR_TIME = 16 384 time samples.
NT = NR_PKTS * NR_TIME  # 16 384
V_A = float(NT * SIG[0][0] ** 2)  # FPGA-0 autocorr = 262 144
V_B = float(NT * SIG[1][0] ** 2)  # FPGA-1 autocorr =  65 536
V_CROSS = float(NT * SIG[0][0] * SIG[1][0])  # cross  = 131 072
VIS_TOL = 0.05  # 5 % relative tolerance (TCC half-precision)

# Per-polarization signal amplitudes for the pol-swap PCAP (same on both FPGAs).
# Using different values per pol slot so XX and YY autocorrs are distinguishable.
SIG_POL0 = (4, 0)  # pol slot 0 (stream 0 within a receiver): real=4, imag=0
SIG_POL1 = (3, 0)  # pol slot 1 (stream 1 within a receiver): real=3, imag=0
V_XX_NORM = float(NT * SIG_POL0[0] ** 2)  # NT×4² = 262,144  (normal: slot0→pol0)
V_YY_NORM = float(NT * SIG_POL1[0] ** 2)  # NT×3² = 147,456  (normal: slot1→pol1)
# When pols are swapped in the SAM (slot0→pol1, slot1→pol0) the roles flip:
V_XX_SWAP = V_YY_NORM  # 147,456
V_YY_SWAP = V_XX_NORM  # 262,144

# ── PCAP generation ───────────────────────────────────────────────────────────


def _global_hdr():
    return struct.pack("IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)


def _pkt_hdr(n):
    return struct.pack("IIII", int(time.time()), 0, n, n)


def _eth():
    return (
        b"\xff\xff\xff\xff\xff\xff"
        + b"\x00\x0a\x95\x9d\x68\x16"
        + struct.pack("!H", 0x0800)
    )


def _ip(src_ip, payload_len):
    total = 20 + 8 + payload_len
    return struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        total,
        0,
        0,
        18,
        17,
        0,
        socket.inet_aton(src_ip),
        socket.inet_aton("192.168.1.255"),
    )


def _udp(payload_len):
    return struct.pack("!HHHH", 36001, 36001, 8 + payload_len, 0)


def _custom_hdr(sample_count, fpga_id, freq_channel):
    # C struct CustomHeader: uint64 sample_count, uint32 fpga_id,
    #                        uint16 freq_channel, uint8[8] padding
    return struct.pack("<QIHQ", sample_count, fpga_id, freq_channel, 0)


def _build_samples(fpga_id):
    # PacketDataStructure: complex<int8_t>[NR_TIME][NR_RECV_PER_PKT][NR_POL]
    # recv 0 → SIG[fpga_id]; others → (0, 0).
    sig = struct.pack("bb", *SIG[fpga_id])  # signed int8 (real, imag)
    zero = b"\x00\x00"
    out = bytearray()
    for _ in range(NR_TIME):
        for r in range(NR_RECV_PER_PKT):
            s = sig if r == 0 else zero
            for _ in range(NR_POL):
                out.extend(s)
    return bytes(out)


def _scales():
    # PacketScaleStructure: int16_t[NR_RECV_PER_PKT][NR_POL], all = 1
    n = NR_RECV_PER_PKT * NR_POL
    return struct.pack(f"<{n}H", *([1] * n))


def build_pcap(path: Path) -> None:
    """
    Write 256 rounds × 8 channels × 2 FPGAs = 4 096 packets.
    This fills exactly one NR_PACKETS_FOR_CORRELATION (=256) correlation block.
    The FPGA ID is encoded in the source IP's third octet so that
    OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET=true reads the correct ID.
    """
    scales_bytes = _scales()
    sample_cache = {fid: _build_samples(fid) for fid in FPGA_IDS}
    zero_samples = b"\x00" * (NR_TIME * NR_RECV_PER_PKT * NR_POL * 2)

    frames = []

    def _write_round(rnd: int, signal: bool) -> None:
        sc = rnd * NR_TIME
        for ch in range(NR_CHAN):
            for fid in FPGA_IDS:
                udp_payload = (
                    _custom_hdr(sc, fid, ch)
                    + scales_bytes
                    + (sample_cache[fid] if signal else zero_samples)
                )
                src_ip = f"10.0.{fid}.10"
                frame = (
                    _eth()
                    + _ip(src_ip, len(udp_payload))
                    + _udp(len(udp_payload))
                    + udp_payload
                )
                frames.append(frame)

    # 256 signal rounds (rnd 1..256, sample_count 64..16384)
    for rnd in range(1, NR_PKTS + 1):
        _write_round(rnd, signal=True)

    # One extra zero-signal round (rnd 257, sample_count 16448) so that
    # latest_packet_received[ch][fpga] >= end_seq + NR_BETWEEN_SAMPLES/2 = 16416,
    # triggering check_buffer_completion() to declare the block done.
    _write_round(NR_PKTS + 1, signal=False)

    with open(path, "wb") as f:
        f.write(_global_hdr())
        for frame in frames:
            f.write(_pkt_hdr(len(frame)))
            f.write(frame)

    total = len(frames)
    size_kb = path.stat().st_size // 1024
    print(f"  PCAP: {total} packets, {size_kb} KB → {path.name}")


def _build_samples_perpol(fpga_id):
    """Samples for the pol-swap PCAP: pol slot 0 gets SIG_POL0, slot 1 gets SIG_POL1."""
    zero = b"\x00\x00"
    out = bytearray()
    pol_sigs = [struct.pack("bb", *SIG_POL0), struct.pack("bb", *SIG_POL1)]
    for _ in range(NR_TIME):
        for r in range(NR_RECV_PER_PKT):
            for p in range(NR_POL):
                out.extend(pol_sigs[p] if r == 0 else zero)
    return bytes(out)


def build_pcap_polswap(path: Path) -> None:
    """
    Like build_pcap but recv 0 pol slot 0 carries amplitude 4 and pol slot 1
    carries amplitude 3 (same on both FPGAs).  Used by the pol-swap tests.
    """
    scales_bytes = _scales()
    sample_cache = {fid: _build_samples_perpol(fid) for fid in FPGA_IDS}
    zero_samples = b"\x00" * (NR_TIME * NR_RECV_PER_PKT * NR_POL * 2)

    frames = []

    def _write_round(rnd: int, signal: bool) -> None:
        sc = rnd * NR_TIME
        for ch in range(NR_CHAN):
            for fid in FPGA_IDS:
                udp_payload = (
                    _custom_hdr(sc, fid, ch)
                    + scales_bytes
                    + (sample_cache[fid] if signal else zero_samples)
                )
                src_ip = f"10.0.{fid}.10"
                frame = (
                    _eth()
                    + _ip(src_ip, len(udp_payload))
                    + _udp(len(udp_payload))
                    + udp_payload
                )
                frames.append(frame)

    for rnd in range(1, NR_PKTS + 1):
        _write_round(rnd, signal=True)
    _write_round(NR_PKTS + 1, signal=False)

    with open(path, "wb") as f:
        f.write(_global_hdr())
        for frame in frames:
            f.write(_pkt_hdr(len(frame)))
            f.write(frame)

    total = len(frames)
    size_kb = path.stat().st_size // 1024
    print(f"  PCAP: {total} packets, {size_kb} KB → {path.name}")


# ── Stream-antenna-map ────────────────────────────────────────────────────────


def make_sam(fpga0_ant: int, fpga1_ant: int) -> dict:
    """
    Minimal map: recv 0 of each FPGA gets the specified antenna ID (both pols).
    All other streams are absent → treated as disconnected by build_permutation().

    Stream numbering: stream k = receiver (k // NR_POL), pol_slot (k % NR_POL).
    So recv 0 occupies streams 0 (pol X) and 1 (pol Y).
    """
    return {
        "fpgas": {
            "0": {
                "streams": [
                    {"stream": 0, "antenna_id": fpga0_ant, "polarization": 0},
                    {"stream": 1, "antenna_id": fpga0_ant, "polarization": 1},
                ]
            },
            "1": {
                "streams": [
                    {"stream": 0, "antenna_id": fpga1_ant, "polarization": 0},
                    {"stream": 1, "antenna_id": fpga1_ant, "polarization": 1},
                ]
            },
        }
    }


def make_sam_disconnected(fpga0_ant: int) -> dict:
    """
    Only FPGA 0 is present in the SAM; FPGA 1's key is entirely absent.
    All 10 receivers of FPGA 1 are therefore disconnected — the pipeline
    should zero all baselines that involve any of those receiver slots.
    """
    return {
        "fpgas": {
            "0": {
                "streams": [
                    {"stream": 0, "antenna_id": fpga0_ant, "polarization": 0},
                    {"stream": 1, "antenna_id": fpga0_ant, "polarization": 1},
                ]
            },
        }
    }


def make_sam_polswap(fpga0_ant: int, fpga1_ant: int) -> dict:
    """
    Same antenna assignments as make_sam but polarizations 0 and 1 are
    exchanged: stream 0 (which carries pol-slot-0 data) is assigned to
    polarization 1, and stream 1 (pol-slot-1 data) is assigned to
    polarization 0.  This should swap XX and YY autocorrelation values.
    """
    return {
        "fpgas": {
            "0": {
                "streams": [
                    {"stream": 0, "antenna_id": fpga0_ant, "polarization": 1},
                    {"stream": 1, "antenna_id": fpga0_ant, "polarization": 0},
                ]
            },
            "1": {
                "streams": [
                    {"stream": 0, "antenna_id": fpga1_ant, "polarization": 1},
                    {"stream": 1, "antenna_id": fpga1_ant, "polarization": 0},
                ]
            },
        }
    }


# ── Baseline index ────────────────────────────────────────────────────────────


def bl(i: int, j: int) -> int:
    """Packed lower-triangular baseline index for canonical receivers i ≤ j."""
    assert i <= j, f"bl() requires i ≤ j, got i={i} j={j}"
    return j * (j + 1) // 2 + i


# ── HDF5 checker ─────────────────────────────────────────────────────────────


def _near(got, expected):
    return abs(got - expected) <= max(1.0, abs(expected) * VIS_TOL)


def check_hdf5(hdf5_path: Path, fpga0_ant: int, fpga1_ant: int, label: str) -> bool:
    """
    Verify the HDF5 visibilities match the expected permutation result.

    Canonical ordering is ascending antenna ID.  If fpga0_ant < fpga1_ant:
      canonical-0 = ant(fpga0)  → signal V_A  (FPGA-0, value 4+0j)
      canonical-1 = ant(fpga1)  → signal V_B  (FPGA-1, value 2+0j)
    Otherwise the assignment flips.
    """
    if fpga0_ant < fpga1_ant:
        ant0, ant1 = fpga0_ant, fpga1_ant  # canonical 0, 1
        v00, v11 = V_A, V_B  # FPGA-0 signal on canonical-0
    else:
        ant0, ant1 = fpga1_ant, fpga0_ant  # canonical 0 = FPGA-1
        v00, v11 = V_B, V_A

    failures = []

    try:
        with h5py.File(hdf5_path, "r") as f:
            # Metadata datasets
            ant_ids = f["antenna_ids"][:]  # [NR_RECEIVERS]
            bl_ants = f["baseline_antenna_ids"][:]  # [NR_BASELINES, 2]
            bl_ids = f["baseline_ids"][:]  # [NR_BASELINES]
            bl_zeroed = f["baseline_zeroed"][:]  # [NR_BASELINES]
            # Visibility data: [N_blocks, NR_CHAN, NR_BASELINES, NR_POL, NR_POL, 2]
            vis = f["visibilities"][:]
    except Exception as exc:
        failures.append(f"HDF5 read error: {exc}")
        _report(label, failures)
        return False

    if vis.shape[0] == 0:
        failures.append("visibilities dataset is empty (no blocks written)")
        _report(label, failures)
        return False

    # channel-0, XX polarization, real part, first block
    vis_re = vis[0, 0, :, 0, 0, 0]  # shape [NR_BASELINES]

    # ── 1. antenna_ids canonical order ───────────────────────────────────────
    # The pipeline must store receiver IDs in ascending-antenna-ID order.
    if int(ant_ids[0]) != ant0 or int(ant_ids[1]) != ant1:
        failures.append(
            f"antenna_ids[0:2] = {list(int(x) for x in ant_ids[:4])}, "
            f"expected [{ant0}, {ant1}, ...] (ascending canonical order)"
        )

    # ── 2. baseline_antenna_ids for non-zero baselines ────────────────────────
    # Each baseline entry records which two antennas it correlates.
    for idx, exp_pair in [
        (bl(0, 0), (ant0, ant0)),
        (bl(0, 1), (ant0, ant1)),
        (bl(1, 1), (ant1, ant1)),
    ]:
        got = (int(bl_ants[idx, 0]), int(bl_ants[idx, 1]))
        if got != exp_pair:
            failures.append(
                f"baseline_antenna_ids[bl={idx}] = {got}, expected {exp_pair}"
            )

    # ── 3. baseline_ids (256*ant_a + ant_b) ─────────────────────────────────
    # Packed integer encoding of the antenna pair for each baseline.
    for idx, exp_id in [
        (bl(0, 0), 256 * ant0 + ant0),
        (bl(0, 1), 256 * ant0 + ant1),
        (bl(1, 1), 256 * ant1 + ant1),
    ]:
        if int(bl_ids[idx]) != exp_id:
            failures.append(
                f"baseline_ids[bl={idx}] = {bl_ids[idx]}, expected {exp_id} "
                f"(= 256×{ant0 if idx != bl(1, 1) else ant1}+{ant0 if idx == bl(0, 0) else ant1})"
            )

    # ── 4. baseline_zeroed: active baselines must be 0 ───────────────────────
    # baseline_zeroed=1 means the pipeline considered a baseline unmapped/inactive.
    for idx in (bl(0, 0), bl(0, 1), bl(1, 1)):
        if int(bl_zeroed[idx]) != 0:
            failures.append(
                f"baseline_zeroed[bl={idx}] = {bl_zeroed[idx]}, expected 0 "
                f"(baseline is active — should not be zeroed)"
            )

    # ── 5. Visibility values ──────────────────────────────────────────────────
    # Checks that the correct signal power lands at each expected baseline.
    # All values are channel-0, XX polarization, real part of the first block.
    got_vals = {}
    for idx, exp_val, name in [
        (bl(0, 0), v00, f"ant{ant0}×ant{ant0} autocorr"),
        (bl(0, 1), V_CROSS, f"ant{ant0}×ant{ant1} cross   "),
        (bl(1, 1), v11, f"ant{ant1}×ant{ant1} autocorr"),
    ]:
        got = float(vis_re[idx])
        got_vals[idx] = got
        if not _near(got, exp_val):
            failures.append(
                f"{name} [bl={idx}]: got {got:.1f}, expected ≈{exp_val:.1f} "
                f"(±{VIS_TOL * 100:.0f}%)"
            )

    # ── 6. All other baselines must be near zero (channel 0, all pols) ───────
    # Any signal in an unexpected baseline means the stream→antenna permutation
    # placed data in the wrong slot.
    ZERO_THRESH = 2.0
    active = {bl(0, 0), bl(0, 1), bl(1, 1)}
    n_nonzero_inactive = 0
    for idx in range(NR_BASELINES):
        if idx in active:
            continue
        chunk = np.abs(vis[0, 0, idx])  # [NR_POL, NR_POL, 2]
        worst = float(chunk.max())
        if worst > ZERO_THRESH:
            n_nonzero_inactive += 1
            failures.append(
                f"inactive bl={idx} (ants {int(bl_ants[idx, 0])},{int(bl_ants[idx, 1])}) "
                f"should be ~0 but max|vis|={worst:.1f} — signal leaked into wrong baseline"
            )

    pass_detail = (
        (
            f"ant_ids=[{ant0},{ant1},...], "
            f"bl({0},{0})={got_vals.get(bl(0, 0), '?'):.0f} "
            f"bl({0},{1})={got_vals.get(bl(0, 1), '?'):.0f} "
            f"bl({1},{1})={got_vals.get(bl(1, 1), '?'):.0f}, "
            f"{NR_BASELINES - 3} other baselines ~0"
        )
        if not failures
        else ""
    )

    _report(label, failures, pass_detail)
    return len(failures) == 0


def _report(label, failures, pass_detail=""):
    if failures:
        print(f"  FAIL {label}:")
        for msg in failures:
            print(f"    · {msg}")
    else:
        suffix = f" — {pass_detail}" if pass_detail else ""
        print(f"  PASS {label}{suffix}")


def check_hdf5_disconnected(hdf5_path: Path, active_ant: int, label: str) -> bool:
    """
    Verify that when FPGA 1 is absent from the SAM:
      - The one active antenna's autocorrelation (bl where both ants = active_ant)
        has baseline_zeroed=0 and visibility ≈ V_A (NT × 4²).
      - Every other baseline has baseline_zeroed=1 (disconnected) and vis ≈ 0.
    """
    failures = []
    try:
        with h5py.File(hdf5_path, "r") as f:
            ant_ids = f["antenna_ids"][:]
            bl_ants = f["baseline_antenna_ids"][:]
            bl_zeroed = f["baseline_zeroed"][:]
            vis = f["visibilities"][:]
    except Exception as exc:
        failures.append(f"HDF5 read error: {exc}")
        _report(label, failures)
        return False

    if vis.shape[0] == 0:
        failures.append("visibilities dataset is empty (no blocks written)")
        _report(label, failures)
        return False

    # Locate the canonical receiver index that was assigned active_ant.
    active_idx = np.where(ant_ids == active_ant)[0]
    if len(active_idx) == 0:
        failures.append(
            f"antenna_ids does not contain ant{active_ant}; "
            f"first 4 values: {list(int(x) for x in ant_ids[:4])}"
        )
        _report(label, failures)
        return False
    c = int(active_idx[0])
    active_bl = bl(c, c)  # autocorrelation baseline for the active receiver

    # 1. Active autocorr must NOT be zeroed.
    if int(bl_zeroed[active_bl]) != 0:
        failures.append(
            f"baseline_zeroed[bl={active_bl}] (ant{active_ant}×ant{active_ant}) = "
            f"{bl_zeroed[active_bl]}, expected 0 — active baseline should not be zeroed"
        )

    # 2. Every other baseline must be zeroed (FPGA 1 is disconnected).
    wrong_zeroed = [
        (idx, int(bl_ants[idx, 0]), int(bl_ants[idx, 1]))
        for idx in range(NR_BASELINES)
        if idx != active_bl and int(bl_zeroed[idx]) != 1
    ]
    if wrong_zeroed:
        shown = wrong_zeroed[:3]
        for idx, a0, a1 in shown:
            failures.append(
                f"baseline_zeroed[bl={idx}] (ants {a0},{a1}) = {bl_zeroed[idx]}, "
                f"expected 1 — this receiver slot is disconnected (no SAM entry)"
            )
        if len(wrong_zeroed) > 3:
            failures.append(
                f"  ... and {len(wrong_zeroed) - 3} more baselines with wrong baseline_zeroed"
            )

    # 3. Active autocorr visibility value (XX pol, real, channel 0, block 0).
    vis_re_xx = vis[0, 0, :, 0, 0, 0]
    got = float(vis_re_xx[active_bl])
    if not _near(got, V_A):
        failures.append(
            f"ant{active_ant}×ant{active_ant} XX autocorr: got {got:.1f}, "
            f"expected ≈{V_A:.1f} (±{VIS_TOL * 100:.0f}%)"
        )

    # 4. All disconnected baselines must have vis ≈ 0 (all pols, all channels).
    ZERO_THRESH = 2.0
    nonzero = [
        (idx, float(np.abs(vis[0, 0, idx]).max()))
        for idx in range(NR_BASELINES)
        if idx != active_bl and float(np.abs(vis[0, 0, idx]).max()) > ZERO_THRESH
    ]
    if nonzero:
        shown = nonzero[:3]
        for idx, worst in shown:
            failures.append(
                f"disconnected bl={idx} (ants {int(bl_ants[idx, 0])},{int(bl_ants[idx, 1])}) "
                f"max|vis|={worst:.1f}, expected ~0 — signal leaked from disconnected FPGA 1"
            )
        if len(nonzero) > 3:
            failures.append(
                f"  ... and {len(nonzero) - 3} more non-zero disconnected baselines"
            )

    n_correctly_zeroed = int(np.sum(bl_zeroed == 1))
    pass_detail = (
        (
            f"ant{active_ant} at canonical idx {c}, "
            f"bl({c},{c}) XX autocorr={got:.0f} ≈ {V_A:.0f}, "
            f"{n_correctly_zeroed}/{NR_BASELINES - 1} disconnected baselines have baseline_zeroed=1 and vis≈0"
        )
        if not failures
        else ""
    )
    _report(label, failures, pass_detail)
    return len(failures) == 0


def check_hdf5_polswap(
    hdf5_path: Path, fpga0_ant: int, fpga1_ant: int, swapped: bool, label: str
) -> bool:
    """
    Verify XX and YY autocorrelations reflect the per-polarization signal
    amplitudes after the SAM maps pol slots to physical polarizations.

    PCAP (build_pcap_polswap): recv 0 pol-slot 0 = amplitude 4,
                                recv 0 pol-slot 1 = amplitude 3 (same on both FPGAs).

    Normal SAM  (slot0→pol0, slot1→pol1):
      canonical pol 0 gets slot-0 data (amp 4) → XX autocorr = NT×4² = 262,144
      canonical pol 1 gets slot-1 data (amp 3) → YY autocorr = NT×3² = 147,456

    Swapped SAM (slot0→pol1, slot1→pol0):
      canonical pol 0 gets slot-1 data (amp 3) → XX autocorr = NT×3² = 147,456
      canonical pol 1 gets slot-0 data (amp 4) → YY autocorr = NT×4² = 262,144
    """
    exp_xx = V_XX_SWAP if swapped else V_XX_NORM
    exp_yy = V_YY_SWAP if swapped else V_YY_NORM

    failures = []
    try:
        with h5py.File(hdf5_path, "r") as f:
            ant_ids = f["antenna_ids"][:]
            vis = f["visibilities"][:]
    except Exception as exc:
        failures.append(f"HDF5 read error: {exc}")
        _report(label, failures)
        return False

    if vis.shape[0] == 0:
        failures.append("visibilities dataset is empty (no blocks written)")
        _report(label, failures)
        return False

    # Find canonical index of the lower-ID antenna (canonical-0) to get its autocorr.
    ant0 = min(fpga0_ant, fpga1_ant)
    active_idx = np.where(ant_ids == ant0)[0]
    if len(active_idx) == 0:
        failures.append(
            f"antenna_ids does not contain ant{ant0}; "
            f"first 4: {list(int(x) for x in ant_ids[:4])}"
        )
        _report(label, failures)
        return False
    c = int(active_idx[0])
    active_bl = bl(c, c)

    # vis shape: [block, channel, baseline, pol_i, pol_j, re_im]
    got_xx = float(vis[0, 0, active_bl, 0, 0, 0])  # XX real
    got_yy = float(vis[0, 0, active_bl, 1, 1, 0])  # YY real

    if not _near(got_xx, exp_xx):
        failures.append(
            f"ant{ant0} XX autocorr: got {got_xx:.1f}, expected ≈{exp_xx:.1f} "
            f"({'swapped' if swapped else 'normal'} pol assignment, ±{VIS_TOL * 100:.0f}%)"
        )
    if not _near(got_yy, exp_yy):
        failures.append(
            f"ant{ant0} YY autocorr: got {got_yy:.1f}, expected ≈{exp_yy:.1f} "
            f"({'swapped' if swapped else 'normal'} pol assignment, ±{VIS_TOL * 100:.0f}%)"
        )

    pass_detail = (
        (
            f"ant{ant0} at canonical idx {c}: XX={got_xx:.0f} (exp≈{exp_xx:.0f}), "
            f"YY={got_yy:.0f} (exp≈{exp_yy:.0f}) — "
            f"{'XX↔YY correctly swapped vs normal' if swapped else 'XX>YY as expected for normal pol order'}"
        )
        if not failures
        else ""
    )
    _report(label, failures, pass_detail)
    return len(failures) == 0


# ── observe runner ────────────────────────────────────────────────────────────


def run_observe(
    observe_bin: Path,
    pcap: Path,
    sam: Path,
    vis_out: Path,
    workdir: Path,
    timeout_secs: int,
) -> None:
    """Invoke observe_2_8; raises RuntimeError if the binary exits non-zero."""
    # +1 extra round of trigger packets (see build_pcap) to push
    # latest_packet_received past the completion threshold.
    n_packets = (NR_PKTS + 1) * NR_CHAN * NR_FPGAS  # = 4112

    cmd = [
        str(observe_bin),
        "-p",
        str(pcap),
        "-v",
        str(vis_out),
        "-f",
        "0",
        "-i",
        "0,1",
        "--stream-antenna-map",
        str(sam),
        "--accumulation-length",
        "1",
        "-n",
        str(n_packets),  # exit once all packets are received
        f"--obs-length={timeout_secs}",  # safety fallback
        "-c",
        str(workdir / "config.json"),
        "-g",
        str(workdir / "weights.json"),
        "-y",
        str(workdir / "alveo_delays.json"),
        "--eigenvalue-num-filename",
        str(workdir / "nr-signal-eigenvalues.json"),
    ]
    print(f"  $ {' '.join(cmd)}")

    # Run in a temp working dir so eigendata_*.hdf5, app.log, output_timings.csv
    # land there rather than polluting the workspace root.
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_secs + 30,  # subprocess hard kill after binary's own limit
        cwd=str(vis_out.parent),  # tmpdir, all paths above are absolute
    )
    if result.returncode != 0:
        print(f"  observe exited {result.returncode}")
        if result.stderr:
            print(f"  stderr (last 3000 chars):\n{result.stderr[-3000:]}")
        raise RuntimeError(f"observe binary exited {result.returncode}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--observe-bin",
        default="bin/observe_2_8",
        help="path to observe_2_8 binary  [default: bin/observe_2_8]",
    )
    ap.add_argument(
        "--workdir",
        default=".",
        help="directory with config.json, weights.json, etc.  [default: .]",
    )
    ap.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="keep temp dir after test (for manual inspection)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="per-run timeout in seconds  [default: 120]",
    )
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    observe_bin = Path(args.observe_bin)
    if not observe_bin.is_absolute():
        observe_bin = (workdir / observe_bin).resolve()

    # Preflight checks
    if not observe_bin.exists():
        print(f"ERROR: binary not found: {observe_bin}", file=sys.stderr)
        return 1
    for req in (
        "config.json",
        "weights.json",
        "alveo_delays.json",
        "nr-signal-eigenvalues.json",
    ):
        if not (workdir / req).exists():
            print(f"ERROR: {req} not found in {workdir}", file=sys.stderr)
            return 1

    tmpdir = Path(tempfile.mkdtemp(prefix="obs_perm_test_"))

    # ── Header ──────────────────────────────────────────────────────────────
    print("=" * 72)
    print("Stream-antenna-map permutation integration test")
    print("=" * 72)
    print(f"Binary  : {observe_bin}")
    print(
        f"Config  : NR_FPGA_SOURCES={NR_FPGAS}, NR_CHANNELS={NR_CHAN}, "
        f"NR_RECEIVERS_PER_PACKET={NR_RECV_PER_PKT}"
    )
    print(f"Tmpdir  : {tmpdir}")
    print()
    print("What this test checks:")
    print("  The pipeline reads a stream-antenna-map (SAM) JSON that assigns")
    print("  each FPGA output stream a physical antenna ID. It then reorders")
    print("  the streams into ascending-antenna-ID (canonical) order before")
    print("  correlating. This test verifies that visibilities land at the")
    print("  correct baseline slots with correct values regardless of which")
    print("  FPGA carries which antenna.")
    print()
    print("Signal injected into PCAP (same PCAP reused across all test cases):")
    print(f"  FPGA 0, stream 0 (recv 0 of 10): int8 (4+0j) for all {NR_CHAN} channels")
    print(f"  FPGA 1, stream 0 (recv 0 of 10): int8 (2+0j) for all {NR_CHAN} channels")
    print(f"  All other receiver slots        : (0+0j)")
    print(
        f"  NT = {NR_PKTS} packets × {NR_TIME} time steps = {NT:,} total time samples"
    )
    print(f"  Expected autocorr (FPGA 0 side) = NT × 4² = {V_A:,.0f}")
    print(f"  Expected autocorr (FPGA 1 side) = NT × 2² = {V_B:,.0f}")
    print(f"  Expected cross-correlation      = NT × 4 × 2 = {V_CROSS:,.0f}")
    print(f"  Tolerance: ±{VIS_TOL * 100:.0f}% (TCC half-precision rounding)")
    print()

    # Test cases: (label, fpga0_ant, fpga1_ant)
    ## Fix this.
    test_cases = [
        ("canonical", "FPGA0→ant10, FPGA1→ant20", 10, 20),
        ("swapped", "FPGA0→ant20, FPGA1→ant10", 20, 10),
        ("other IDs", "FPGA0→ant5,  FPGA1→ant15", 5, 15),
    ]

    total_tests = len(test_cases) + 3
    try:
        # Build the shared test PCAP
        pcap_path = tmpdir / "test_input.pcap"
        n_signal_pkts = NR_PKTS * NR_CHAN * NR_FPGAS
        n_trigger_pkts = NR_CHAN * NR_FPGAS
        print(
            f"Building test PCAP  ({n_signal_pkts} signal packets + "
            f"{n_trigger_pkts} trigger packets)..."
        )
        build_pcap(pcap_path)
        print(
            f"  Trigger round: sample_count={NR_PKTS * NR_TIME + NR_TIME} "
            f"(pushes latest_packet_received past the buffer-completion threshold)"
        )
        print()

        all_pass = True
        for case_num, (kind, mapping, fpga0_ant, fpga1_ant) in enumerate(test_cases, 1):
            ant0 = min(fpga0_ant, fpga1_ant)
            ant1 = max(fpga0_ant, fpga1_ant)
            canon0_fpga = 0 if fpga0_ant < fpga1_ant else 1
            v00 = V_A if fpga0_ant < fpga1_ant else V_B
            v11 = V_B if fpga0_ant < fpga1_ant else V_A

            print("-" * 72)
            print(f"Test {case_num}/{total_tests}: {kind}  ({mapping})")
            print("-" * 72)
            print(f"  Stream-antenna-map:")
            print(f"    FPGA 0, stream 0 (recv 0, both pols) → antenna ID {fpga0_ant}")
            print(f"    FPGA 1, stream 0 (recv 0, both pols) → antenna ID {fpga1_ant}")
            print(
                f"  Canonical order after sorting by antenna ID: [ant{ant0}, ant{ant1}]"
            )
            print(
                f"    canonical receiver 0 = ant{ant0}  ← carried by FPGA {canon0_fpga} "
                f"(signal 4+0j)"
                if canon0_fpga == 0
                else f"    canonical receiver 0 = ant{ant0}  ← carried by FPGA {canon0_fpga} "
                f"(signal 2+0j)"
            )
            print(
                f"    canonical receiver 1 = ant{ant1}  ← carried by FPGA {1 - canon0_fpga}"
            )
            print(f"  Expected visibilities (channel 0, XX pol, real, block 0):")
            print(f"    bl(0,0) = ant{ant0}×ant{ant0} autocorr  → {v00:>9,.0f}")
            print(f"    bl(0,1) = ant{ant0}×ant{ant1} cross      → {V_CROSS:>9,.0f}")
            print(f"    bl(1,1) = ant{ant1}×ant{ant1} autocorr  → {v11:>9,.0f}")
            print(f"    bl(i,j) = j*(j+1)/2 + i  (packed lower-triangular index)")
            print(
                f"    all {NR_BASELINES - 3} other baselines (out of {NR_BASELINES} total) → ~0"
            )
            print(f"  Checks: antenna_ids order, baseline_antenna_ids, baseline_ids,")
            print(
                f"          baseline_zeroed flags, visibility values, inactive baselines ~0"
            )

            sam_path = tmpdir / f"sam_{fpga0_ant}_{fpga1_ant}.json"
            vis_path = tmpdir / f"vis_{fpga0_ant}_{fpga1_ant}.hdf5"
            sam_path.write_text(json.dumps(make_sam(fpga0_ant, fpga1_ant), indent=2))

            label = f"{kind}: {mapping}"
            try:
                run_observe(
                    observe_bin, pcap_path, sam_path, vis_path, workdir, args.timeout
                )
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                print(f"  FAIL: {exc}")
                all_pass = False
                continue

            if not vis_path.exists():
                print(f"  FAIL: HDF5 not written at {vis_path}")
                all_pass = False
                continue

            if not check_hdf5(vis_path, fpga0_ant, fpga1_ant, label):
                all_pass = False

        # ── Test 4: disconnected stream ──────────────────────────────────────
        print("-" * 72)
        print(
            f"Test {len(test_cases) + 1}/{total_tests}: disconnected  (FPGA0→ant10, FPGA1 absent from SAM)"
        )
        print("-" * 72)
        print("  Stream-antenna-map:")
        print("    FPGA 0, stream 0 (recv 0, both pols) → antenna ID 10")
        print("    FPGA 1: NOT present in the SAM JSON — all 10 of its receiver slots")
        print("            are treated as disconnected (no physical antenna attached)")
        print("  Expected outcome:")
        print(
            f"    bl(c,c) where ant_ids[c]=10: baseline_zeroed=0, XX autocorr ≈ {V_A:,.0f}"
        )
        print(f"    all other {NR_BASELINES - 1} baselines: baseline_zeroed=1, vis ≈ 0")
        print(
            "  Checks: baseline_zeroed=0 for the one active autocorr, baseline_zeroed=1"
        )
        print("          for all baselines involving FPGA-1's disconnected receivers,")
        print("          visibility values confirm signal only at the active baseline")

        disc_sam = tmpdir / "sam_disconnected.json"
        disc_vis = tmpdir / "vis_disconnected.hdf5"
        disc_sam.write_text(json.dumps(make_sam_disconnected(10), indent=2))
        try:
            run_observe(
                observe_bin, pcap_path, disc_sam, disc_vis, workdir, args.timeout
            )
            if disc_vis.exists():
                if not check_hdf5_disconnected(
                    disc_vis, 10, "disconnected: FPGA0→ant10, FPGA1 absent"
                ):
                    all_pass = False
            else:
                print(f"  FAIL: HDF5 not written at {disc_vis}")
                all_pass = False
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"  FAIL: {exc}")
            all_pass = False

        # ── Tests 5 & 6: polarization swap ──────────────────────────────────
        print()
        polswap_pcap = tmpdir / "polswap_input.pcap"
        print(
            f"Building pol-swap PCAP  ({NR_PKTS * NR_CHAN * NR_FPGAS} signal + "
            f"{NR_CHAN * NR_FPGAS} trigger packets)..."
        )
        print("  Same packet layout as the main PCAP, but recv 0 now has:")
        print(
            f"    pol slot 0 (stream 0) = int8 ({SIG_POL0[0]}+{SIG_POL0[1]}j)  "
            f"→ expected autocorr when mapped to pol 0: NT×{SIG_POL0[0]}² = {V_XX_NORM:,.0f}"
        )
        print(
            f"    pol slot 1 (stream 1) = int8 ({SIG_POL1[0]}+{SIG_POL1[1]}j)  "
            f"→ expected autocorr when mapped to pol 1: NT×{SIG_POL1[0]}² = {V_YY_NORM:,.0f}"
        )
        build_pcap_polswap(polswap_pcap)
        print()

        for case_num, swapped in [
            (len(test_cases) + 2, False),
            (len(test_cases) + 3, True),
        ]:
            kind = "pol-swapped" if swapped else "pol-normal"
            sam_fn = make_sam_polswap if swapped else make_sam
            exp_xx = V_XX_SWAP if swapped else V_XX_NORM
            exp_yy = V_YY_SWAP if swapped else V_YY_NORM
            print("-" * 72)
            print(
                f"Test {case_num}/{total_tests}: {kind}  (FPGA0→ant10, FPGA1→ant20, per-pol signals)"
            )
            print("-" * 72)
            print("  Stream-antenna-map:")
            if swapped:
                print(
                    "    FPGA 0/1, stream 0 (carries pol-slot-0 data, amplitude 4) → polarization 1"
                )
                print(
                    "    FPGA 0/1, stream 1 (carries pol-slot-1 data, amplitude 3) → polarization 0"
                )
                print("  Pol swap: slot-0 data (amp 4) is routed to canonical pol 1,")
                print("            slot-1 data (amp 3) is routed to canonical pol 0")
            else:
                print(
                    "    FPGA 0/1, stream 0 (carries pol-slot-0 data, amplitude 4) → polarization 0"
                )
                print(
                    "    FPGA 0/1, stream 1 (carries pol-slot-1 data, amplitude 3) → polarization 1"
                )
                print(
                    "  Normal: slot-0 data (amp 4) → canonical pol 0, slot-1 (amp 3) → pol 1"
                )
            print(f"  Expected autocorrelations for ant10 (channel 0, block 0):")
            print(
                f"    XX (pol0×pol0) = NT × {SIG_POL1[0] if swapped else SIG_POL0[0]}² "
                f"= {exp_xx:>9,.0f}"
            )
            print(
                f"    YY (pol1×pol1) = NT × {SIG_POL0[0] if swapped else SIG_POL1[0]}² "
                f"= {exp_yy:>9,.0f}"
            )
            print(
                "  Checks: XX and YY autocorr values at the canonical-0 autocorr baseline"
            )

            sam_path = tmpdir / f"sam_pol{'swap' if swapped else 'norm'}.json"
            vis_path = tmpdir / f"vis_pol{'swap' if swapped else 'norm'}.hdf5"
            sam_path.write_text(json.dumps(sam_fn(10, 20), indent=2))
            label = f"{kind}: FPGA0→ant10, FPGA1→ant20"
            try:
                run_observe(
                    observe_bin, polswap_pcap, sam_path, vis_path, workdir, args.timeout
                )
                if vis_path.exists():
                    if not check_hdf5_polswap(vis_path, 10, 20, swapped, label):
                        all_pass = False
                else:
                    print(f"  FAIL: HDF5 not written at {vis_path}")
                    all_pass = False
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                print(f"  FAIL: {exc}")
                all_pass = False

        print()
        print("=" * 72)
        if all_pass:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")
        print("=" * 72)
        return 0 if all_pass else 1

    finally:
        if args.keep_artifacts:
            print(f"Artifacts retained at: {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
