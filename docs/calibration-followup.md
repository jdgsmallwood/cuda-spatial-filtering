# Calibration follow-up

## FPGA delay confidence

Deferred until after the observing run, as requested.

- Solve on **eight coarse channels**, then reuse the sample offsets when
  changing band or increasing to forty channels. Retuning alone does not
  require a new delay solution; a reboot or verified timing change does.
- The eight-channel verification reproduced all six established pair delays
  with zero closure residual. FPGA 0–2 had 173/960 votes (18.0%). Agreement
  across all pairs and independent captures is the current acceptance evidence;
  the vote fraction is not a statistical confidence level.
- Equal peak voting currently counts weak, contaminated, and strongly coherent
  antenna/channel/hand combinations equally. The combinations share data and
  cannot be treated as independent Bernoulli trials.
- Report the strongest competing peaks, peak sharpness, usable sample overlap,
  packet completeness, and support separately by hand and frequency.
- Test repeatability on separate short capture windows and clean channel
  subsets. Do not address an inadequate window by collecting a large
  forty-channel PCAP.
- Develop meaningful rejection tests with missing packets, narrowband
  interference, different host capture start times, and a deliberately wrong
  pairwise peak. Exact graph closure is necessary but cannot by itself rule out
  a common systematic error.

## Calibration phases

- Keep a smooth instrumental phase model separate from frequency flagging.
  Smoothing cannot make interference or nonfinite visibilities usable.
- Include the measured phase ramp that repeats across each coarse channel;
  a single polynomial over the whole band misses this structure.
- Validate entire withheld coarse channels, withheld baselines, and a separate
  later capture before extending the observing mask.
- Investigate the origin of the repeating phase structure and the weaker XX
  sky-model agreement after the observation. Neither is yet evidence of a
  specific GPU correction bug.
- A 225–256 MHz before/after comparison spanning an intervening eight-channel
  retune showed large phase changes shared by antennas on the same FPGA.
  Refitting the later capture recovered good model agreement, while the
  eight-channel sample-delay solution remained unchanged. Calibrate after the
  final channel setup and validate without retuning. Isolate the retune-related
  phase mechanism later; do not mistake it for a changed integer sample delay.
