# Pulsar signal chain audit

## Scope

The pre-observation audit inspected reception, packet ownership, signed FPGA
delay correction, channelization, calibration and steering, and both DSPSR
input formats. The original observation executables were preserved, with
their hashes and capture capabilities checked. After a later live voltage
failure, separately built candidates were authorized and tested as described
below.

## Executed checks

| Existing test executable | Passed | Relevant coverage |
| --- | ---: | --- |
| ProcessorTests | 19 | Four FPGA producers, delayed commits, ring wrap, future-packet horizon, large signed offsets, FPGA ID mapping |
| IbverbsPlacementTests | 4 | Direct placement versus normal packet placement, delay offsets, rollover, stale packets |
| StreamReorderTests | 6 | Per-FPGA packet scales, cross-FPGA receptor mapping, gains and delay corrections |
| FineChannelizerTests | 6 | Tone frequency/order, edge trimming, filter input and output packing |
| FineChannelizedPulsarFoldTests | 6 | Voltage beam computation, intensity detection and repeated-boundary removal |
| FineChannelizedStarspinTests | 6 | Four-FPGA cross phases, fine-bin ordering, broadband sample-delay correction |
| PointingTests | 8 | Coordinate conventions and fine-channel frequency grid |

All 55 existing tests passed. A subsequent candidate build also passed all
23 tests in the affected suites: 9 fine-channelized pulsar, 6 channelizer and
8 stream reorder tests, including the new regression cases below.

The retained live four-FPGA sample capture was also inspected. Packet scales
were between 1 and 16; no scale had its unsigned high bit set. A bounded sample
of approximately 82 million real/imaginary components had maximum absolute
value 888 after packet scaling, below the finite FP16 limit of 65,504.
This checks the input conversion; it does not bound every later filter output.

## Data layout and corrections

1. A LAMBDA payload has a 22-byte custom header, 20 two-byte scales and
   `64 × 10 × 2 × 2` signed one-byte sample components. Total: 2,622 bytes.
2. Host sample storage is `[coarse][packet][FPGA][time][receiver][pol][IQ]`.
   Packet scales use the flattened FPGA/receiver index. Gains at this stage
   use `[coarse][pol][receiver]`.
3. Each FPGA counter offset is split into a nearest whole-packet multiple
   and a signed within-packet remainder. Packet placement handles the former;
   the GPU gather handles the latter using the preceding/following packet
   padding. This happens before cross-FPGA receptor reordering.
4. Pulsar-fold and starspin share these unpack, gather, reorder and
   channelizer routines. Their outputs diverge at beamforming versus
   correlation.
5. For the fine-calibrated tracked beam, the measured gain is inverted once
   and folded into the steering weight. The raw-sample gain table stays unity.
   The physical receptor map must match the calibration's polarization and
   antenna identity conventions. All three GPU buffers receive steering
   refreshes, and the updated weights are permuted before each beam operation.
6. Forty coarse channels produce 1,120 retained processing bins. Removing
   the repeated first bin of each coarse channel after the first produces
   1,081 uniformly spaced DSPSR channels.
7. Voltage output is `[time][frequency][pol][IQ]`, two polarizations, two
   components, float32. Intensity output is `[time][frequency]`, with
   `I = |X|² + |Y|²`. These match the local DSPSR float unpacker's input layouts.
   Both folds request summed intensity with `-d 1`.

At the production shape, one block contains 512 fine-channel time samples:
8,855,552 bytes of voltage or 2,213,888 bytes of intensity. The sample interval
is 34.56 microseconds. Its reciprocal is the 28.935185 kHz frequency spacing.
Header frequencies were independently compared against the calibration mask,
including the first and last bins; block sizes and pointing metadata matched.

## Additional coverage built and executed

The following regression cases were added, built separately and passed on
the GPU, preserving the original observation executables and capabilities:

- Four-FPGA unpacking with distinct values across channels, packets, time,
  receptors, polarizations, packet scales and complex gains.
- Four-FPGA sample gathers with signed offsets, including both half-packet
  boundaries, checked against a CPU indexing oracle.
- Fine-channelized pulsar beams with independent broadband FPGA offsets,
  compared sample-by-sample with an undelayed reference and a wrong-delay control.
- Production 40-channel boundary compaction across multiple times, both
  polarizations and multiple beams; voltage values and summed intensity are
  checked independently.
- A single strong receptor in a four-FPGA, 32-bin channelizer case: all beam
  values must remain finite, restored float voltage must exceed the FP16
  limit, and normalized squared gain-linearity error must be below 1e-6.

Existing reorder tests use identical signals in both polarizations in some
cases, and existing pulsar compaction tests use only one time and one beam.
The additional cases exercise those previously missing dimensions.

## FP16 output overflow and correction

A later voltage capture produced infinities and NaNs in its GPU output ring
before DSPSR read it. One receptor's scaled packet components reached 15,232,
which fits FP16; however, a CPU transcription of the installed filter and FFT
predicted output components up to 738,276. FP16's maximum finite value is
65,504. Checking only the unpacked input range was therefore insufficient.

The pulsar candidate applies voltage scaling by 1/64 through gpu-filter's
existing float output scale **before** narrowing to FP16. Beam output is
restored by 64 in its float permutation before intensity detection or voltage
export. The optional visibility path restores power by 4096. Both factors
are exact binary powers; 1/64 also survives the dependency's six-decimal
kernel-option serialization exactly. This is headroom for the measured case,
not a guarantee against arbitrary input levels.

The 60-second live candidate control used the original gain solution. All
13,283,328 sampled scalar ring values were finite, DSPSR issued no corruption
warnings, and all seven archived subintegrations retained the 263 selected
channels. The problematic polarization nevertheless had about 978 times the
other polarization's beam power in the retained band. A subsequent science
observation excludes that receptor alone; this is a data-quality flag and
does not establish the physical cause of the strong input.

That subsequent GPU voltage observation completed 600 seconds with zero
DSPSR corruption warnings. All 60 subintegrations retained 254–263 channels
after the calibration mask and additional RFI flagging. Exported profile
values, scales and offsets were finite. Three subintegrations containing
strong zero-DM transients were subsequently excluded in a separate copy,
leaving 570 seconds and reducing the zero-DM diagnostic score from 23.40
to 3.15. A weak expected-DM feature remains unconfirmed. This validates
the measured numerical failure's correction, not a pulsar detection.

## Known limitations

- An absolute frequency-label check remains open. The shifted 32-point FFT
  puts baseband DC at untrimmed bin 16 (retained bin 14). The current frequency
  helper instead labels retained bins symmetrically about the coarse centre,
  assigning bin 14 a frequency half a fine bin above that centre. If the
  configured coarse frequency is the true baseband centre, all labels are
  high by 14.467593 kHz. The header/mask agreement above establishes internal
  consistency, not absolute RF accuracy. For the observing geometry this
  would produce at most 0.595 degrees of differential steering phase and
  approximately 116 microseconds of residual dispersion across the retained
  band at DM 56.7109. This is about 0.88 of a 256-bin Crab phase bin. A known
  RF tone is the definitive hardware check. No unilateral header or mask
  shift was made before observing; an eventual correction must keep all
  frequency metadata and calibration conventions consistent.
- The 16-tap channelizer zero-pads its trailing look-ahead region at every
  block. Up to 15 of 512 fine-channel time samples are affected by an edge
  transient: 2.93% of the block. This is not a measured 2.93% sensitivity loss.
- `UTC_START` is stamped before GPU/DSPSR startup, rather than at the first
  received sample. One smoke run had about a 19-second difference. Absolute
  pulse phase and absolute timestamp accuracy are therefore not established.
  Steering is separately refreshed from the current wall clock. A fixed
  start-time offset does not itself change the stream's sample interval.
- Packet scale storage is signed in the GPU path and unsigned in the offline
  sample analyzer. Captured scales do not reach the differing range; handling
  values above 32,767 is not validated by this observation.
- One voltage smoke archive reported a corrupt fine channel already outside
  the calibration mask. No conclusion about its underlying cause is made.
  Masking removes that channel from the detection analysis, but is not a fix
  for the corruption.
- Calibration and the ephemeris remain empirical limitations. Passing software
  tests does not demonstrate a pulsar detection or establish sensitivity.

## Fold diagnostics

`scripts/analyze_pulsar_fold.py` reads a dedispersed Stokes-I PSRFITS archive
and makes pulse, time/phase, frequency/phase and trial-DM plots. It preserves
the channel weights and reports consistency at a fixed candidate phase in
independent time and frequency halves. A synthetic dispersed fold recovered
its injected DM of 62.5, checking the sign and units of the residual DM shifts.

The script maximizes diagnostic scores over phase and pulse width; the DM
curve adds further trials. These scores are not trial-corrected detection
significances. A detection decision requires inspecting repeatability and
dispersion behavior, not merely selecting the highest plotted score.

```sh
pam -D -a PSRFITS -e fits crab.masked.rfi.ar
python3 scripts/analyze_pulsar_fold.py crab.masked.rfi.fits \
  --output-prefix fold_diagnostics
```
