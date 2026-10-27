# Starspin–DiFX correlation verification

This workflow compares the coarse-only Starspin executable with **DiFX 2.9.0's
`mark5access/zerocorr`**, an independent FFTW-based correlation utility distributed
with DiFX. It does **not** yet test the distributed `mpifxcorr` application,
geometric delay models, or Starspin's fine PFB. The existing NumPy PFB reference
remains required. No telescope or FPGA is accessed by this workflow.

## Build the independent reference

Use DiFX tag `v2.9.0`, commit `39401aa88b24d3084fe683f52d23975a7b5da3d4`.
Build its unmodified mark5access package. Prerequisites are a C compiler,
autoconf, automake, libtool, pkg-config, FFTW3 development files, and Git.
Python needs the repository's NumPy, h5py, and pytest dependencies.
Matplotlib generates the summary plot.

```bash
git clone --depth 1 --branch v2.9.0 https://github.com/difx/difx.git /tmp/starspin-difx-src
cd /tmp/starspin-difx-src/libraries/mark5access
autoreconf -fi
./configure --prefix=/tmp/starspin-difx-install
make -j4
make install
```

From the Starspin repository, first run the small external compatibility test:

```bash
DIFX_ZEROCORR=/tmp/starspin-difx-install/bin/zerocorr \
  python3 -m pytest -q tests/test_difx_visibility_reference.py
```

The external test uses a known quarter-turn phase, unequal sample values, and a
nonzero file offset. It checks the complex sign, absolute power, and full-band
normalization against analytic expectations. Without `DIFX_ZEROCORR`, that one
test is explicitly skipped; portable tests still run.

## Run the dataset through both packages

In the [reviewer build environment](reviewer-guide.md), configure normally and
build the existing coarse target:

```bash
cmake -S . -B build_difx -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build_difx --parallel 2 --target starspin_1_8_coarse
python3 scripts/check_difx_visibility_parity.py \
  --zerocorr /tmp/starspin-difx-install/bin/zerocorr \
  --difx-source /tmp/starspin-difx-src \
  --binary build_difx/apps/starspin_1_8_coarse \
  --output-dir build/difx-validation
```

Choose the CUDA architecture appropriate to your GPU. If using Docker, mount
the DiFX source/install directories at those same paths, or build DiFX inside
the container. Output directories must be new and contain no whitespace.
There are 1,760 small DiFX jobs per integration; allow several minutes.

Omit `--binary` to run only the CPU compatibility gate. Its JSON contains
`starspin_vs_difx_passed: null`; it is never evidence of a GPU pass. Use
`--integrations 1` for an initial smoke test; the default requires at least ten
complete integrations for the paper run.

The runner generates seeded, quantized shared noise with distinct complex gains
and independent receiver noise on all 20 input streams. Eight coarse channels
and ten dual-polarization receivers exercise all 55 packed baselines and XX,
XY, YX, YY products. Two extra blocks accommodate startup and EOF. The PCAP is
finite, not looped. Every GPU integration must be complete, within the fixture,
and nonoverlapping; all emitted integrations are compared using sample counters.

An independently implemented CODIF decoder checks the PCAP against the saved
canonical integer array before converting it to VDIF. Conversion preserves
every integer sample; clipping, requantization, incomplete frames, missing
packets, and unsupported channel geometry are rejected. This initial fixture
uses one FPGA, unity gains, and zero delays, so it does not exercise inter-board
delay correction.

## Conventions and acceptance

Starspin accumulates `x_i,p * conj(x_j,q)`. Each DiFX input contains one complex
8-bit channel with 64 samples per frame, offset-binary IQ, and a 32-byte VDIF
header. `VDIFC_128-390625m27-1-8` represents the exact sample rate `25000000/27`
Hz without rounding it to an integer MHz rate. Synthetic epoch is 2000-01-01.
Physical sky-frequency coordinates are not checked by this zero-delay test.
Each integration is sliced before VDIF packing, preserving the starting sample
in its frame counter. Two zero guard frames accommodate decoder read-ahead;
the requested FFT count excludes them. This avoids a DiFX 2.9 short-file reader
edge case where a seek into the latter half can return premature EOF. The
unmodified DiFX code is used, and a late-interval regression checks this path.

The normalization is derived from the pinned source, not fitted to Starspin:

1. `format_vdif.c` reconstructs 8-bit values as `(byte - 128) / 3.3`.
   The runner removes `SYMMETRIC_8_BIT_VDIF` from the child environment, avoiding
   its alternative half-integer reconstruction convention.
2. `zerocorr.c` scales each complex FFT by `0.5 / FFT_length`, forms
   `FFT(x) * conj(FFT(y))`, and averages the FFT products.
3. Summing **all** 64 spectral bins and multiplying by
   `4 * integration_sample_count * 3.3^2 * 128^2` yields Starspin's accumulated
   voltage units, by Parseval's identity. The last factor restores the fixture's
   CODIF scale exponent 7. No bins are trimmed, no PFB is applied, and no gain
   or phase is fitted. DiFX's text output has seven significant digits.

The runner checks DiFX against NumPy, and (when supplied) Starspin against both.
Every integration must have relative complex RMS below `1e-3` and maximum
autopower-normalized complex error below `2e-3`, matching the existing reviewer
reference thresholds. DiFX's two independently output autopowers must agree
with the input powers within `1e-3`. Conjugation, missing scale, and polarization
swap negative controls must fail. Nonfinite output and incomplete FFT counts
are errors, even when the external program exits successfully.

## Evidence and remaining work

### Local verification, 19 October 2026

The freshly built `starspin_1_8_coarse` passed against unmodified DiFX 2.9.0
`zerocorr` on an RTX 4060, driver 580.178.04, CUDA 12.8.1, Python 3.10.12,
and NumPy 2.2.6. No network or telescope access was enabled during replay.

| Measurement | Result |
| --- | --- |
| Complete integrations | 11, each 16,384 samples |
| Channels / baselines / polarization products | 8 / 55 / 4 |
| Complex visibilities compared | 19,360 |
| Maximum Starspin–DiFX relative RMS | `2.3708038444005157e-8` |
| Maximum autopower-normalized error | `9.246423825285717e-8` |
| Maximum strong-product phase error | `3.3514207314713884e-6` degrees |
| Starspin–NumPy complex residual | Exactly zero for this integer fixture |
| Missing packets in compared integrations | 0 |
| Conjugation / missing-scale / polarization-swap controls | All rejected |
| Targeted Python tests, including actual DiFX | 15 passed |

The integration range was `[1024, 181248)` in sample counters; initial replay
startup samples were excluded by the actual HDF5 integration metadata. The
PCAP SHA-256 is `116c12c46438c7aad43e07ac0cd2e87b4e684873c44bc389fd6f1f1cacad38bf`.
The fixture contains 196,608 samples per stream, including startup/EOF margin.

Local evidence is in `build/difx-gpu-validation/`; build logs, dependency
revisions, package inventories, and container identity are in `build_difx/`.
The container image ID is
`sha256:de79942bd069631cc0ed00c40b053089c2b8ee0f2be46578b55bd205646c32bd`.
This used the current working tree and the previously documented local TCC
revision `36e5edc3bc49772670edb5451f7b397342d26602`, not a clean public-clone
rehearsal. The dependency/release gate in [JOSS readiness](joss-readiness.md)
remains open. The new CI job is prepared but has not run on GitHub.

### Archive and coverage

Retain `fixture.json`, canonical IQ, PCAP, all VDIF files, DiFX configurations,
logs and spectra, `provenance.json`, Starspin command/log/HDF5/audit files,
`difx_visibilities.npz`, `comparison.png`, and `comparison.json`. Archive the exact source tree,
including uncommitted changes and submodule revisions, and the build/container
identity. Binary hashes alone do not identify their shared libraries. Have a
colleague reproduce the public fixture before claiming independent reproduction.

Follow-on coverage:

- Repeat with `mpifxcorr`, matching the verified complex-input scaling, zero
  differential delay/rate, integration boundaries, and all spectral bins.
- Inject known inter-board packet delays and compare Starspin's alignment with
  independently aligned DiFX input. Do not reuse this zero-delay fixture's
  settings for live observations.
- Compare a suitable recorded observation only after verifying boot state,
  tuning, the boot-matched delay file, and delay-graph closure. If unverifiable,
  reboot selected FPGAs, run full eight-channel `init-tune`, solve delays and
  confirm closure before widening the band. Read back every board's channel
  range before capture and preserve those records. Do not reboot between
  captures within the same verified session just for a binary/channel change.
- Compare fine channels only after matching filter coefficients, retained bins,
  rounding, and block-boundary padding. A coarse-only pass does not validate
  that signal path, packet-loss weighting, real-time throughput, or calibration.

References: [DiFX source](https://github.com/difx/difx/tree/v2.9.0/libraries/mark5access),
[zerocorr configuration/output](https://difx.readthedocs.io/en/latest/sources/difx_files.html#zerocorr),
[Starspin output conventions](starspin-output.md).
