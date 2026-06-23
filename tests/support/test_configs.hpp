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

} // namespace test_support
