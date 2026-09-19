#pragma once

#include "spatial/packet_formats.hpp"

// Canonical small `LambdaConfig` instantiations shared across pipeline tests.
//
// These intentionally mirror the smallest configs already proven to drive a
// real GPU pipeline correctly end to end (test_pipeline.cu's
// `Config`/`MultiFPGAConfig`), so that exact-value golden checks such as
// test_pipeline.cu::Ex1's expected 8.0f/64.0f results keep working when ported
// onto the shared harness in `pipeline_harness.hpp`.
namespace test_support {

// 1 channel, 1 FPGA source, 4 receivers -- the smallest layout that still
// satisfies the Tensor Core Correlator's constraints (NR_RECEIVERS padded to a
// multiple of 32, NR_POLARIZATIONS == 2; see CLAUDE.md "Domain gotchas").
using SmallSingleFPGAConfig =
    LambdaConfig<1,     // NR_CHANNELS
                 1,     // NR_FPGA_SOURCES
                 8,     // NR_TIME_STEPS_PER_PACKET
                 4,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 4,     // NR_RECEIVERS_PER_PACKET
                 1,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 10000  // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 >;

// Same overall sizing, but the receivers are spread across 3 FPGA sources (2
// receivers/packet each) -- exercises FPGA-to-FPGA delay alignment and
// multi-source reassembly paths that SmallSingleFPGAConfig can't reach.
using SmallMultiFPGAConfig =
    LambdaConfig<1,     // NR_CHANNELS
                 3,     // NR_FPGA_SOURCES
                 8,     // NR_TIME_STEPS_PER_PACKET
                 6,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 2,     // NR_RECEIVERS_PER_PACKET
                 1,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 10000  // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 >;

// 2-channel variant -- exercises multi-channel output independence: tests can
// feed distinct data per channel and assert on per-channel beam/visibility
// output without cross-channel contamination.
using SmallTwoChannelConfig =
    LambdaConfig<2,     // NR_CHANNELS
                 1,     // NR_FPGA_SOURCES
                 8,     // NR_TIME_STEPS_PER_PACKET
                 4,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 4,     // NR_RECEIVERS_PER_PACKET
                 1,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 10000  // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 >;

// 2-packet variant -- exercises multi-packet accumulation: with constant
// (2,-2) input and scale=1 the correlator integrates over 2*8=16 time steps,
// giving autocorrelation power 128 (double the single-packet value of 64).
using SmallTwoPacketConfig =
    LambdaConfig<1,     // NR_CHANNELS
                 1,     // NR_FPGA_SOURCES
                 8,     // NR_TIME_STEPS_PER_PACKET
                 4,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 4,     // NR_RECEIVERS_PER_PACKET
                 2,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 10000  // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 >;

// Fine-channelized config: 1 coarse/FPGA channel split into 32 fine channels (NR_FINE_CHANNELS,
// trailing template arg) -- matches the project's production default (NR_OBSERVING_FINE_CHANNELS=32).
// NR_TIME_STEPS_PER_PACKET=512 so NR_TIME_STEPS_FOR_CORRELATION=512 divides evenly by
// NR_FINE_CHANNELS=32 into NR_SAMPLES_PER_FINE_CHANNEL=16 -- the minimum satisfying
// FineChannelizer's two internal constraints (a multiple of gpu-filter's fixed
// NR_TIMES_PER_ITERATION=16, and of NR_TIMES_PER_OUTPUT_BLOCK=8). NR_FINE_CHANNEL_EDGE_TRIM=2
// matches the project's production default (NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM=2), so this
// config's effective processing channel count (Config::NR_CHANNELS) is 32 - 2*2 = 28, not 32 --
// don't hardcode 32 against Config::NR_CHANNELS anywhere, always derive it.
using SmallFineChannelizedConfig =
    LambdaConfig<1,     // NR_CHANNELS (coarse/FPGA)
                 1,     // NR_FPGA_SOURCES
                 512,   // NR_TIME_STEPS_PER_PACKET
                 4,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 4,     // NR_RECEIVERS_PER_PACKET
                 1,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 10000, // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 false, // OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET
                 128,   // FFT_DOWNSAMPLE_FACTOR
                 32,    // NR_FINE_CHANNELS
                 2      // NR_FINE_CHANNEL_EDGE_TRIM
                 >;

// Fine-channelized config sized for pipelines with no correlation/beamforming (antenna spectra) --
// deliberately smaller NR_FINE_CHANNELS=8 (vs. the production-matching 32 above) so
// MultiChannelAntennaFFTOutputType's trailing (NR_TIME_STEPS_PER_FINE_CHANNEL / FFT_DOWNSAMPLE_FACTOR)
// axis works out to a small positive value (16/4=4) rather than truncating to 0 the way the
// production FFT_DOWNSAMPLE_FACTOR=128 would for a 16-sample fine channel. NR_FINE_CHANNEL_EDGE_TRIM=1
// (not 2) since NR_FINE_CHANNELS=8 here is much smaller than the production default of 32; effective
// processing channel count (Config::NR_CHANNELS) is 8 - 2*1 = 6.
using SmallFineChannelizedAntennaSpectraConfig =
    LambdaConfig<1,     // NR_CHANNELS (coarse/FPGA)
                 1,     // NR_FPGA_SOURCES
                 128,   // NR_TIME_STEPS_PER_PACKET
                 4,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 4,     // NR_RECEIVERS_PER_PACKET
                 1,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 10000, // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 false, // OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET
                 4,     // FFT_DOWNSAMPLE_FACTOR
                 8,     // NR_FINE_CHANNELS
                 1      // NR_FINE_CHANNEL_EDGE_TRIM
                 >;

// Same dimensions as SmallFineChannelizedAntennaSpectraConfig, but with channelization disabled
// (NR_FINE_CHANNELS=1, NR_FINE_CHANNEL_EDGE_TRIM=0) -- for regression-testing
// LambdaAntennaSpectraPipeline's original whole-band-FFT path, which must stay byte-for-byte
// unaffected by the NR_FINE_CHANNELS>1 wiring added alongside it.
using SmallAntennaSpectraConfig =
    LambdaConfig<1,     // NR_CHANNELS (coarse/FPGA)
                 1,     // NR_FPGA_SOURCES
                 128,   // NR_TIME_STEPS_PER_PACKET
                 4,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 4,     // NR_RECEIVERS_PER_PACKET
                 1,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 10000, // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 false, // OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET
                 4      // FFT_DOWNSAMPLE_FACTOR
                 >;

// Same shape as the antenna-spectra configs above, for LambdaBeamformedSpectraPipeline (which
// beamforms, unlike LambdaAntennaSpectraPipeline).
using SmallFineChannelizedBeamformedSpectraConfig =
    LambdaConfig<1,     // NR_CHANNELS (coarse/FPGA)
                 1,     // NR_FPGA_SOURCES
                 128,   // NR_TIME_STEPS_PER_PACKET
                 4,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 4,     // NR_RECEIVERS_PER_PACKET
                 1,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 10000, // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 false, // OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET
                 4,     // FFT_DOWNSAMPLE_FACTOR
                 8,     // NR_FINE_CHANNELS
                 1      // NR_FINE_CHANNEL_EDGE_TRIM
                 >;

// Disabled-channelization counterpart, for regression-testing the original whole-band-FFT path.
using SmallBeamformedSpectraConfig =
    LambdaConfig<1,     // NR_CHANNELS (coarse/FPGA)
                 1,     // NR_FPGA_SOURCES
                 128,   // NR_TIME_STEPS_PER_PACKET
                 4,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 4,     // NR_RECEIVERS_PER_PACKET
                 1,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 10000, // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 false, // OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET
                 4      // FFT_DOWNSAMPLE_FACTOR
                 >;

// For LambdaProjectionPipeline (correlate + eigendecompose + projection-accumulate, no
// beamforming, no FFT) -- same small fine-channelized shape as the spectra configs above.
using SmallFineChannelizedProjectionConfig =
    LambdaConfig<1,     // NR_CHANNELS (coarse/FPGA)
                 1,     // NR_FPGA_SOURCES
                 128,   // NR_TIME_STEPS_PER_PACKET
                 4,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 4,     // NR_RECEIVERS_PER_PACKET
                 1,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 10000, // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 false, // OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET
                 128,   // FFT_DOWNSAMPLE_FACTOR (unused by this pipeline)
                 8,     // NR_FINE_CHANNELS
                 1      // NR_FINE_CHANNEL_EDGE_TRIM
                 >;

// Disabled-channelization counterpart, for regression-testing the original correlate+eigen path.
using SmallProjectionConfig =
    LambdaConfig<1,     // NR_CHANNELS (coarse/FPGA)
                 1,     // NR_FPGA_SOURCES
                 128,   // NR_TIME_STEPS_PER_PACKET
                 4,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 4,     // NR_RECEIVERS_PER_PACKET
                 1,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 10000, // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 false  // OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET
                 >;

// For LambdaAdaptiveBeamformedSpectraPipeline: correlate + eigendecompose + RFI-mitigate +
// dual-beam beamform, whole-band FFT deleted when channelized.
using SmallFineChannelizedAdaptiveConfig =
    LambdaConfig<1,     // NR_CHANNELS (coarse/FPGA)
                 1,     // NR_FPGA_SOURCES
                 128,   // NR_TIME_STEPS_PER_PACKET
                 4,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 4,     // NR_RECEIVERS_PER_PACKET
                 1,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 1,     // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 false, // OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET
                 128,   // FFT_DOWNSAMPLE_FACTOR (unused once channelized)
                 8,     // NR_FINE_CHANNELS
                 1      // NR_FINE_CHANNEL_EDGE_TRIM
                 >;

// Disabled-channelization counterpart -- NR_TIME_STEPS_PER_PACKET=64 to match the pipeline's
// hardcoded CUFFT_FFT_SIZE=64 for the (still-active) whole-band FFT path.
using SmallAdaptiveConfig =
    LambdaConfig<1,   // NR_CHANNELS (coarse/FPGA)
                 1,   // NR_FPGA_SOURCES
                 64,  // NR_TIME_STEPS_PER_PACKET (= CUFFT_FFT_SIZE)
                 4,   // NR_RECEIVERS
                 2,   // NR_POLARIZATIONS
                 4,   // NR_RECEIVERS_PER_PACKET
                 1,   // NR_PACKETS_FOR_CORRELATION
                 1,   // NR_BEAMS
                 32,  // NR_PADDED_RECEIVERS
                 32,  // NR_PADDED_RECEIVERS_PER_BLOCK
                 1    // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 >;

// For LambdaCorrBeamOnlyGPUPipeline (benchmark-only: correlate + beamform, no eigen, no FFT) --
// same small fine-channelized shape as the other pipeline configs above.
using SmallFineChannelizedCorrBeamConfig =
    LambdaConfig<1,     // NR_CHANNELS (coarse/FPGA)
                 1,     // NR_FPGA_SOURCES
                 128,   // NR_TIME_STEPS_PER_PACKET
                 4,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 4,     // NR_RECEIVERS_PER_PACKET
                 1,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 1,     // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 false, // OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET
                 128,   // FFT_DOWNSAMPLE_FACTOR (unused by this pipeline)
                 8,     // NR_FINE_CHANNELS
                 1      // NR_FINE_CHANNEL_EDGE_TRIM
                 >;

// For LambdaPulsarFoldPipeline (delay/reorder -> beamform, with an optional
// correlate+eigendecompose+project RFI-mitigation path) -- same small
// fine-channelized shape as the other pipeline configs above.
using SmallFineChannelizedPulsarFoldConfig =
    LambdaConfig<1,     // NR_CHANNELS (coarse/FPGA)
                 1,     // NR_FPGA_SOURCES
                 128,   // NR_TIME_STEPS_PER_PACKET
                 4,     // NR_RECEIVERS
                 2,     // NR_POLARIZATIONS
                 4,     // NR_RECEIVERS_PER_PACKET
                 1,     // NR_PACKETS_FOR_CORRELATION
                 1,     // NR_BEAMS
                 32,    // NR_PADDED_RECEIVERS
                 32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                 1,     // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 false, // OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET
                 128,   // FFT_DOWNSAMPLE_FACTOR (unused by this pipeline)
                 8,     // NR_FINE_CHANNELS
                 1      // NR_FINE_CHANNEL_EDGE_TRIM
                 >;

} // namespace test_support
