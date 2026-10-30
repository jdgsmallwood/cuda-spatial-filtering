# Beamforming method comparison with two GPU buffers

## Methods

All methods were compared with two GPU buffers on an NVIDIA RTX 4000 Ada
(20 GiB), using CUDA 12.8.93 and a PCIe Gen 3 ×16 connection under load. Synthetic
input comprised 40 receiver slots, two polarizations and 40 coarse channels.
Each block contained 16,384 time samples per coarse channel and 104,857,600 bytes
of useful complex int8 payload. A 32-bin polyphase filter bank retained 28 bins
per coarse channel, producing 1,120 fine channels with 512 samples per channel.
Paired original and filtered beam outputs were formed for each polarization.
The weights-only baseline used identical weights for both outputs. Adaptive
modes nulled rank two, with rank-one spatial protection enabled or disabled;
the protected GPU benchmark used a uniform synthetic signature.

Each eager/graph configuration was measured three times over 192 blocks after
warmup, with graph instantiation outside timing. The selected 30 trials come
from the supplied randomized, interleaved matrix. All selected runs succeeded;
all 15 graph trials recorded two captured graphs. Throughput is decimal GB/s of
useful input payload, including H2D and GPU computation but excluding packet
parsing and output writing. Values are medians with observed min/max ranges,
not confidence intervals. Buffer count is fixed rather than selected separately
for each method.

## Results

With two buffers, graph-enabled median throughput was 11.872 GB/s for weights-only,
9.508 GB/s for unprotected coarse decomposition, and 9.407 GB/s for protected
coarse decomposition. Fine decomposition reached 3.871 GB/s without protection
and 3.718 GB/s with protection. Matched eager medians were 11.863, 9.477, 9.379,
3.875 and 3.725 GB/s, respectively. Graph/eager median differences were below
0.4% in every case; this fixed-buffer comparison shows no substantial throughput
benefit from CUDA Graphs. The method-level throughput differences are much larger
than the measured effect of graph execution at this buffer count.

## Figure caption

**Two-buffer beamforming throughput.** Median input payload throughput for
weights-only, coarse and fine decomposition, with two GPU buffers in every case.
Grey and blue bars denote eager and CUDA Graph execution. Error bars show the
observed minimum and maximum of three 192-block trials; small ranges may be hidden
by the markers. Adaptive modes null rank two; protected configurations preserve
a rank-one synthetic spatial signature. Timing includes H2D transfers and GPU
compute and excludes packet parsing and output writing.

## Scope and verification

This replaces the earlier best-across-buffer-counts comparison for the current
paper presentation. Other buffer counts are excluded. The supplied kernel profiles
and full-application replays used three buffers and are omitted from this package;
they cannot substantiate a two-buffer timing breakdown or replay result.

The source archive manifest, selected trial counts, capture counts and byte/time
arithmetic were checked. Raw logs, traces and HDF5 files were absent from the
compact archive, so the checks establish internal consistency of supplied records,
not independent reproduction. No live RFI suppression, solar retention or Starspin
correlation-only benchmark is included. Source revision and dependency provenance
remain those of the supplied measurement archive; its hash is recorded in
`input_provenance.json`. Private deployment identifiers are excluded.
