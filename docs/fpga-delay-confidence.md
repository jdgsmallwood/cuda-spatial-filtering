# Validate FPGA delays without retuning

`capture_delay_samples` reuses `LibibverbsPacketCapture` and its receive-into-ring
path. It drains the current stream, retains selected coarse channels in bounded
RAM, stops reception, and then writes packet payloads. It does not run a
correlator, tune the instrument, or change a delay file. Run it between
observations; do not attach competing receive queues during an observation.

## Build and capture

```bash
cmake --build build --target capture_delay_samples delay_correlation -j 4
sudo setcap cap_net_raw+ep build/apps/capture_delay_samples

# Supply four interfaces in FPGA index order and the existing delay solution.
build/apps/capture_delay_samples \
  --interfaces "$FPGA_INTERFACES" --seed-delays delay-seed.json \
  --channels 296,297,298,299,300,301,302,303 \
  --samples 524288 --output build/delay-check
```

`SPATIAL_CAPTURE_CPUS` optionally pins the four receive threads. Channel selectors
are examples: they must be present in the instrument's existing output. Between
1 and 40 selectors can be retained. The sample limit is 524,288 per channel
(about 0.566 seconds at the instrument's sample interval). Eight channels retain
about 0.69 GB across four FPGAs; forty retain about 3.44 GB. Memory is allocated
before reception, and capture has a finite timeout. Disk speed cannot block the
receive loop. An existing output directory is refused.

The seed offsets establish a common sample-counter interval. Every FPGA must
reach its end. The saved payloads retain the original counters and scale factors.
A separate analyzer reconstructs samples, including sub-packet offsets, counts
duplicates and missing samples, and rejects insufficient coverage. Receiving an
expected total packet count alone is not used as proof of completeness.

The shared receiver cleanup now destroys the flow/queue before deregistering
memory, then releases the protection domain and device. Closing the device
before deregistering memory caused a sampler crash after the first capture had
already been saved. A subsequent live capture exited normally with the corrected
order. Existing observing executables need not be rebuilt to use this tool.

## Analyze and compare channel selections

```bash
python scripts/analyze_delay_samples.py build/delay-check \
  --stream-map stream_antenna_map.json --output build/delay-check/result.json

# Optional GPU acceleration, numerically tested against the CPU calculation:
python scripts/analyze_delay_samples.py build/delay-check \
  --stream-map stream_antenna_map.json \
  --gpu-library build/apps/libdelay_correlation.so \
  --output build/delay-check/result-gpu.json
```

The CPU implementation uses NumPy/SciPy; the optional shared library uses the
project's CUDA/cuFFT stack. GPU analysis should also run between observations.
Use `--channels` to compare subsets of a wider saved capture without recapturing
or retuning. The native stream map supplies connected same-hand receptors;
antenna 14 is excluded by default and can be changed with `--exclude-antennas`.
A whole-antenna identity swap does not change the sample lag; receptor hand swaps
must still be represented in the map.

### Algorithm

1. Select a common, counter-aligned interval using the seed. Missing samples stay
   zero, with an explicit coverage mask. Duplicate packet counters are counted.
2. Split into 32 non-overlapping windows of 16,384 samples by default. Only windows
   with at least 99.5% coverage on every FPGA are used for a channel. Require at
   least four usable windows in each independent half.
3. Remove DC, FFT each receptor, form all live same-hand cross-spectra for each
   FPGA pair, and average complex cross-spectra over windows. Unknown receiver
   phases do not need to be equal: each baseline is averaged separately.
4. Inverse FFT with zero padding for linear correlation. Search residual lags
   over ±1,024 samples by default, rather than forcing the answer to the seed.
   Normalize by receptor energies and the nominal overlap at each lag. Small
   retained gaps can perturb this normalization; coverage is reported explicitly.
5. Combine lag power over baselines/channels after normalization to each
   baseline's median lag power. Clip each contribution at 100 to limit domination
   by a single interferer. This is a practical diagnostic score, not a likelihood.
6. Solve all six measured edges for four integer offsets (reference FPGA 1).
   Repeat on each time half and after leaving out each coarse channel.

`validated` requires exact closure, identical offsets in both time halves,
identical lag choices after omitting any channel, no peak at a search boundary,
and separation from other integer peaks. The thresholds are 1.5 for a competitor
more than four samples away and 1.1 for the strongest other integer lag, including
neighbors. These thresholds are explicit heuristics, not calibrated confidence
probabilities.

The analyzer reports all-baseline peak votes, and separately votes from peaks
whose power exceeds twenty times their median lag power. The latter is a
conditional strong-signal subset; it must not be presented as the agreement of
all data. Baselines share antennas, so their votes are not independent Bernoulli
trials. Likewise, a peak measured in median-absolute-deviation units is not a
Gaussian detection significance.

This validates an existing solution locally. It does not replace the initial
wide delay search after a reboot, and it cannot detect an alternative outside
its explicit search range. A broader capture or search should be used when the
seed is uncertain. Stationary correlated interference can also produce a stable
peak: compare clean frequency subsets and retain the sky/calibration checks.
More channels may add interference rather than confidence.

Results are diagnostic JSON and NPZ files. No operational delay file is ever
written or replaced automatically.

## Tests

```bash
python -m pytest -q tests/test_delay_samples.py
SPATIAL_DELAY_GPU_LIBRARY=build/apps/libdelay_correlation.so \
  python -m pytest -q tests/test_delay_samples.py
```

Tests cover partial packets, gaps, duplicate counters, nonzero residual signs,
six-edge closure, competing peaks, complete packet-file analysis with deliberately
wrong seeds, and optional numerical agreement of GPU and CPU complex correlations.
