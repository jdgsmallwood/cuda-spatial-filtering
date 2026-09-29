# Starspin GPU payload traffic at 40 coarse channels

The four-FPGA production shape has 40 raw receiver slots, 36 connected
antennas, two polarizations, 256 packets per correlation block, 64 time samples
per packet, and a 32-way/16-tap fine PFB. After dropping two fine channels on
each edge, TCC correlates 1,120 channels. The measured live cadence used here
is 56.513 blocks/s. The RTX 4000 Ada Generation specifies 360 GB/s of memory
bandwidth.

Starspin now reads the raw packet buffer in a fused preparation kernel that
applies packet scales and gains, FPGA sample delays, and the canonical stream
map, then writes the PFB's complex-float input. It preserves the original
half-rounding step before writing that input. The PFB processes only the 36
active receivers. The old half packet, permuted, aligned, and reordered buffers
are not allocated or accessed on the fine-channelized production path.

| Stage | Payload read MB/block | Payload write MB/block | GB/s at live cadence |
|---|---:|---:|---:|
| Fused raw packet to PFB input | 94.372 | 388.547 | 27.29 |
| Fine PFB, repeated source loads | 731.382 | 188.744 | 52.00 |
| Gather retained channels into TCC layout | 165.151 | 165.151 | 18.67 |
| TCC, three 32-receiver tiles | 495.452 | 23.869 | 29.35 |
| Restore visibility scale | 23.869 | 23.869 | 2.70 |
| Accumulate visibilities | 47.739 | 23.869 | 4.05 |
| **Total** | **1,557.965** | **814.049** | **134.05** |

All MB are decimal. The fused read is
`40 × 256 × 64 × 36 × 2 polarizations × 2 bytes/complex-int8`.
Its write includes 15 × 32 zero look-ahead samples per receiver:
`40 × 36 × 2 × (16,384 + 480) × 8 bytes/complex-float`.
The PFB's 16-output-time iteration requests 31 input times for every 32 FFT
positions; this counts overlapping loads repeatedly. TCC's three receiver-pair
tiles likewise request some receiver samples repeatedly. Its output has
`1,120 × (36 × 37 / 2) × 4 polarization pairs × 8 bytes/complex-float`
or 23.869 MB per block.

The table is a **payload-buffer access model**, not a measurement of external
DRAM traffic. It excludes reads of small lookup arrays such as scales, gains,
delays, stream permutations, and optional de-ripple gains. Those can be
requested many times by the fused kernel but should be evaluated using actual
cache and DRAM counters. PFB and TCC reuse can also make measured DRAM reads
much smaller than their requested loads. The 134.05 GB/s sum is 37.2% of peak
as a workload equivalent; it is not measured utilization or a performance
claim.

For comparison, the pre-fusion 40-receiver PFB worksheet estimated 238.09
GB/s of kernel payload requests, or 66.1% peak equivalent. The modelled
reduction is about 104 GB/s of requested payload accesses. Host-to-device
input is still 107.328 MB/block (about 6.07 GB/s at the live cadence), and
visibility download remains about 23.869 MB once per 56 integrated blocks.
Measure `dram__bytes_read.sum`, `dram__bytes_write.sum`, and elapsed time on a
production-shape Nsight Compute run to determine the actual DRAM and runtime
change.
