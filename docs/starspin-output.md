# Starspin output conventions

Starspin writes HDF5 visibilities plus mapping and run-audit metadata. The core
writer is `HDF5VisibilitiesWriter` in `include/spatial/writers.hpp`; the application
adds channelization and integration attributes in `apps/starspin.cu`.

| Dataset | Meaning |
| --- | --- |
| `visibilities` | Float32 array `[integration, channel, baseline, polarization 1, polarization 2, real/imaginary]` |
| `vis_seq_nums` | Integer start and end sample counters; end is the start of the final 64-sample packet |
| `vis_missing_nums` | Float32 columns: missing packets, total packets, missing percentage |
| `baseline_receiver_indices` | Explicit receiver-index pair for each packed baseline |
| `baseline_antenna_ids` | Physical antenna IDs corresponding to each receiver pair |
| `audit/run_manifest_json` | Command, normalized configuration, input-file contents and provenance |

Read the baseline arrays rather than guessing physical antenna order. Packed
receiver pairs enumerate `(i,j)` for `j=0..N-1`, `i=0..j`; there are `N(N+1)/2`
baselines. Products use the convention `sum_t x_i,p(t) * conj(x_j,q(t))`, as
checked by `scripts/check_codif_visibility_parity.py`. The last axis is real,
then imaginary. Polarization indices refer to the captured stream map; do not
infer a sky basis or Stokes products from the index alone.

Visibilities are accumulated correlations in scaled input-sample units, not Jy
and not automatically normalized by the integration duration. Gain application
is optional. Missing-packet counts must be consulted before interpreting output;
no automatic correction converts missing samples into a fully sampled
measurement. Count columns are Float32, so large values have limited integer
precision. `mjd_start` is writer construction time, not an authoritative
astronomical timestamp. Use packet timing and sample counters for precision
alignment of real observations.

Fine-channelized Starspin records `fine_fft_channels=32`,
`fine_channels_per_coarse=27`, and `first_raw_fine_bin=3`. Under the LAMBDA
frequency plan, the centre for coarse channel `c` and FFT-shifted bin `k` is
`c * 781250 + (k - 16) * (781250 / 27)` Hz, for `k=3..29`. Channels are ordered
coarse-first, then fine. Use `unique_processing_frequencies_hz` in
`scripts/fine_frequency_axis.py` for this layout. Older 28-bin captures require
their own metadata-aware interpretation. Coarse-only targets have eight output
channels and disable the fine PFB.

The fine PFB zero-pads the look-ahead at each correlation block boundary. This
introduces a boundary transient and is represented in the numerical reference;
it is not a continuous-history filter across blocks. Receiver pruning also
varies by build: four-FPGA targets with at least 40 coarse channels require the
canonical stream map and process 36 connected receivers, producing 666 baselines.

The shared parser accepts options used by companion applications. Starspin does
not produce beams, pulsar folds, eigenvectors, or RFI-projected products. Treat
special-NIC and GPUDirect modes as separate hardware verification configurations.
