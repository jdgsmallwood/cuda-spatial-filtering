// Relocation correctness test for the GPUDirect ingest path (see
// /home/ubuntu/.claude/plans/i-want-to-start-breezy-lampson.md). Exercises
// the real mechanism LibibverbsGpuDirectPacketCapture uses --
// ProcessorState::resolve_gpu_slot() + gpudirect_relocate_kernel -- against a
// real ProcessorState + LambdaGPUPipeline (no ibverbs/NIC hardware needed,
// unlike libibverbs.hpp itself which requires HAVE_IBVERBS and is untestable
// in this environment). Differential design: one harness fills a buffer
// entirely through the reactive path as ground truth; a second harness
// relocates the same packets via resolve_gpu_slot()+the relocation kernel and
// must match byte-for-byte.
#include "spatial/gpudirect_relocate.cuh"
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"
#include "support/pipeline_harness.hpp"
#include "support/synthetic_packets.hpp"

#include <complex>
#include <cstring>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

using Config = LambdaConfig<2,     // NR_CHANNELS
                            2,     // NR_FPGA_SOURCES
                            8,     // NR_TIME_STEPS_PER_PACKET
                            4,     // NR_RECEIVERS
                            2,     // NR_POLARIZATIONS
                            2,     // NR_RECEIVERS_PER_PACKET
                            4,     // NR_PACKETS_FOR_CORRELATION
                            1,     // NR_BEAMS
                            32,    // NR_PADDED_RECEIVERS
                            32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                            10000  // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                            >;
constexpr size_t NR_BUFFERS = 3;

// Deterministic per-packet pattern, used both by the reactive-path ground
// truth fill and by the synthetic landing-pool content relocated via
// resolve_gpu_slot()+the kernel -- if both paths land the same bytes at the
// same offsets, the two runs must match exactly.
std::complex<int8_t> sample_pattern(size_t channel, size_t fpga, int pkt, int t, int r, int p) {
  const int v = static_cast<int>(channel * 17 + fpga * 5 + (pkt + 2) * 3 + t + r + p) % 100 + 1;
  return {static_cast<int8_t>(v), static_cast<int8_t>(-v)};
}
int16_t scale_pattern(size_t channel, size_t fpga, int pkt, int r, int p) {
  return static_cast<int16_t>(channel * 100 + fpga * 10 + (pkt + 2) + r + p + 1);
}

std::vector<uint8_t> build_samples_bytes(size_t channel, size_t fpga, int pkt) {
  typename Config::PacketDataStructure data{};
  for (int t = 0; t < static_cast<int>(Config::NR_TIME_STEPS_PER_PACKET); ++t)
    for (int r = 0; r < static_cast<int>(Config::NR_RECEIVERS_PER_PACKET); ++r)
      for (int p = 0; p < static_cast<int>(Config::NR_POLARIZATIONS); ++p)
        data[t][r][p] = sample_pattern(channel, fpga, pkt, t, r, p);
  std::vector<uint8_t> bytes(sizeof(data));
  std::memcpy(bytes.data(), &data, sizeof(data));
  return bytes;
}
std::vector<uint8_t> build_scales_bytes(size_t channel, size_t fpga, int pkt) {
  typename Config::PacketScaleStructure scales{};
  for (int r = 0; r < static_cast<int>(Config::NR_RECEIVERS_PER_PACKET); ++r)
    for (int p = 0; p < static_cast<int>(Config::NR_POLARIZATIONS); ++p)
      scales[r][p] = scale_pattern(channel, fpga, pkt, r, p);
  std::vector<uint8_t> bytes(sizeof(scales));
  std::memcpy(bytes.data(), &scales, sizeof(scales));
  return bytes;
}

// Relocates one packet's synthetic payload through the real
// gpudirect_relocate_kernel (batch size 1) -- mirrors what
// LibibverbsGpuDirectPacketCapture::resolve_and_stage/relocate_and_repost do
// per packet, minus the ibverbs completion-queue plumbing.
struct RelocationScratch {
  void *landing_samples = nullptr, *landing_scales = nullptr;
  void **src_s = nullptr, **dst_s = nullptr, **src_c = nullptr, **dst_c = nullptr;

  RelocationScratch() {
    CUDA_CHECK(cudaMalloc(&landing_samples, sizeof(typename Config::PacketDataStructure)));
    CUDA_CHECK(cudaMalloc(&landing_scales, sizeof(typename Config::PacketScaleStructure)));
    CUDA_CHECK(cudaHostAlloc((void **)&src_s, sizeof(void *), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **)&dst_s, sizeof(void *), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **)&src_c, sizeof(void *), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **)&dst_c, sizeof(void *), cudaHostAllocMapped));
  }
  ~RelocationScratch() {
    cudaFree(landing_samples);
    cudaFree(landing_scales);
    cudaFreeHost(src_s);
    cudaFreeHost(dst_s);
    cudaFreeHost(src_c);
    cudaFreeHost(dst_c);
  }

  void relocate(const std::vector<uint8_t> &samples_in, const std::vector<uint8_t> &scales_in,
               void *dst_samples_addr, void *dst_scales_addr) {
    CUDA_CHECK(cudaMemcpy(landing_samples, samples_in.data(), samples_in.size(),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(landing_scales, scales_in.data(), scales_in.size(),
                         cudaMemcpyHostToDevice));
    src_s[0] = landing_samples;
    dst_s[0] = dst_samples_addr;
    src_c[0] = landing_scales;
    dst_c[0] = dst_scales_addr;
    gpudirect_relocate_kernel<<<1, 256>>>(src_s, dst_s, static_cast<int>(samples_in.size()));
    gpudirect_relocate_kernel<<<1, 64>>>(src_c, dst_c, static_cast<int>(scales_in.size()));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
};

} // namespace

TEST(GpuDirectScatterTest, RelocationMatchesReactivePathAcrossChannelsAndFpgas) {
  // --- Ground truth: full reactive-path fill of one buffer. ---
  auto weights_ref = test_support::make_unity_beam_weights<Config>();
  auto pipeline_ref =
      test_support::pipeline_factories::make_gpu_pipeline<Config>(NR_BUFFERS, &weights_ref);
  auto output_ref = std::make_shared<SingleHostMemoryOutput<Config>>();
  test_support::SyntheticPipelineRun<Config, NR_BUFFERS> ref_harness(
      *pipeline_ref, output_ref, /*fpga_delays=*/{}, /*fpga_id_map=*/{{0, 0}, {1, 1}});
  ref_harness.run(sample_pattern, scale_pattern, /*start_sample=*/10000);

  const size_t samples_bytes = sizeof(typename Config::InputPacketSamplesType);
  const size_t scales_bytes = sizeof(typename Config::PacketScalesType);
  std::vector<uint8_t> expected_samples(samples_bytes), expected_scales(scales_bytes);
  CUDA_CHECK(cudaMemcpy(expected_samples.data(), pipeline_ref->gpu_landing_samples_ptr(0),
                       samples_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(expected_scales.data(), pipeline_ref->gpu_landing_scales_ptr(0),
                       scales_bytes, cudaMemcpyDeviceToHost));

  // --- GPUDirect path: only the first packet goes through the reactive path
  // (to trigger initialize_buffers()); every other packet is relocated via
  // resolve_gpu_slot() + gpudirect_relocate_kernel. ---
  auto weights_gd = test_support::make_unity_beam_weights<Config>();
  auto pipeline_gd =
      test_support::pipeline_factories::make_gpu_pipeline<Config>(NR_BUFFERS, &weights_gd);
  auto output_gd = std::make_shared<SingleHostMemoryOutput<Config>>();
  test_support::SyntheticPipelineRun<Config, NR_BUFFERS> gd_harness(
      *pipeline_gd, output_gd, /*fpga_delays=*/{}, /*fpga_id_map=*/{{0, 0}, {1, 1}});
  ProcessorStateBase &state = gd_harness.processor_state();

  const uint64_t start_sample = 10000;
  const int NTS = static_cast<int>(Config::NR_TIME_STEPS_PER_PACKET);
  const int NPC = static_cast<int>(Config::NR_PACKETS_FOR_CORRELATION);

  // Only feeds the pinned-host ring (initialize_buffers() trigger) --
  // ingest_and_scale's H2D copy into the pipeline's *device* buffers only
  // happens on buffer completion, which this harness never reaches (that's
  // the whole point: every packet's device-side placement below goes through
  // resolve_gpu_slot() + the relocation kernel instead, including this one).
  test_support::feed_lambda_packet<Config>(
      state, start_sample, /*fpga_id=*/0, /*freq_channel=*/0,
      [&](int t, int r, int p) { return sample_pattern(0, 0, 0, t, r, p); },
      [&](int r, int p) { return scale_pattern(0, 0, 0, r, p); });
  state.process_all_available_packets();

  RelocationScratch scratch;
  for (size_t channel = 0; channel < Config::NR_FPGA_CHANNELS; ++channel) {
    for (size_t fpga = 0; fpga < Config::NR_FPGA_SOURCES; ++fpga) {
      for (int pkt = -1; pkt <= NPC; ++pkt) {
        const uint64_t sample_count = start_sample + static_cast<uint64_t>(pkt) * NTS;
        void *dst_samples = nullptr;
        void *dst_scales = nullptr;
        const bool resolved = state.resolve_gpu_slot(
            sample_count, static_cast<uint32_t>(fpga), static_cast<uint16_t>(channel),
            dst_samples, dst_scales);
        ASSERT_TRUE(resolved) << "channel=" << channel << " fpga=" << fpga << " pkt=" << pkt;

        scratch.relocate(build_samples_bytes(channel, fpga, pkt),
                         build_scales_bytes(channel, fpga, pkt), dst_samples, dst_scales);
      }
    }
  }

  std::vector<uint8_t> actual_samples(samples_bytes), actual_scales(scales_bytes);
  CUDA_CHECK(cudaMemcpy(actual_samples.data(), pipeline_gd->gpu_landing_samples_ptr(0),
                       samples_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(actual_scales.data(), pipeline_gd->gpu_landing_scales_ptr(0),
                       scales_bytes, cudaMemcpyDeviceToHost));

  EXPECT_EQ(actual_samples, expected_samples);
  EXPECT_EQ(actual_scales, expected_scales);
}

TEST(GpuDirectScatterTest, ResolveGpuSlotReturnsFalseForStalePacket) {
  auto weights = test_support::make_unity_beam_weights<Config>();
  auto pipeline = test_support::pipeline_factories::make_gpu_pipeline<Config>(NR_BUFFERS, &weights);
  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();
  test_support::SyntheticPipelineRun<Config, NR_BUFFERS> harness(*pipeline, output);
  ProcessorStateBase &state = harness.processor_state();

  const uint64_t start_sample = 10000;
  test_support::feed_lambda_packet<Config>(
      state, start_sample, 0, 0,
      [&](int t, int r, int p) { return sample_pattern(0, 0, 0, t, r, p); },
      [&](int r, int p) { return scale_pattern(0, 0, 0, r, p); });
  state.process_all_available_packets();

  void *dst_samples = nullptr, *dst_scales = nullptr;
  const uint64_t stale_sample = start_sample - 100 * Config::NR_TIME_STEPS_PER_PACKET;
  EXPECT_FALSE(state.resolve_gpu_slot(stale_sample, 0, 0, dst_samples, dst_scales));
}

// zero_missing_scales_kernel (spatial.cuh) correctness: only (channel,
// packet, fpga) slots with arrivals==false get zeroed.
TEST(GpuDirectScatterTest, ZeroMissingScalesTouchesOnlyMissingSlots) {
  constexpr size_t NR_CHANNELS = 2, NR_PACKETS_PLUS2 = 4, NR_FPGAS = 2;
  constexpr size_t NR_RECEIVERS_PER_PACKET = 2, NR_POLARIZATIONS = 2;
  constexpr size_t NR_RECEIVERS = NR_FPGAS * NR_RECEIVERS_PER_PACKET;
  constexpr size_t NR_SCALE_ELEMS = NR_CHANNELS * NR_PACKETS_PLUS2 * NR_RECEIVERS * NR_POLARIZATIONS;
  constexpr size_t NR_ARRIVAL_ELEMS = NR_CHANNELS * NR_PACKETS_PLUS2 * NR_FPGAS;

  std::vector<int16_t> host_scales(NR_SCALE_ELEMS);
  for (size_t i = 0; i < NR_SCALE_ELEMS; ++i) host_scales[i] = static_cast<int16_t>(i + 1); // all non-zero

  std::vector<bool> missing(NR_ARRIVAL_ELEMS, false);
  missing[0] = true;                       // channel=0 packet=0 fpga=0 -- missing
  missing[NR_ARRIVAL_ELEMS - 1] = true;    // last (channel,packet,fpga) -- missing

  bool *arrivals_host = nullptr;
  CUDA_CHECK(cudaHostAlloc((void **)&arrivals_host, NR_ARRIVAL_ELEMS * sizeof(bool),
                          cudaHostAllocMapped));
  for (size_t i = 0; i < NR_ARRIVAL_ELEMS; ++i) arrivals_host[i] = !missing[i];
  bool *arrivals_dev = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer((void **)&arrivals_dev, arrivals_host, 0));

  int16_t *scales_dev = nullptr;
  CUDA_CHECK(cudaMalloc(&scales_dev, NR_SCALE_ELEMS * sizeof(int16_t)));
  CUDA_CHECK(cudaMemcpy(scales_dev, host_scales.data(), NR_SCALE_ELEMS * sizeof(int16_t),
                       cudaMemcpyHostToDevice));

  zero_missing_scales<NR_CHANNELS, NR_PACKETS_PLUS2, NR_FPGAS, NR_RECEIVERS_PER_PACKET,
                      NR_POLARIZATIONS>(scales_dev, arrivals_dev, /*stream=*/0);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int16_t> result(NR_SCALE_ELEMS);
  CUDA_CHECK(cudaMemcpy(result.data(), scales_dev, NR_SCALE_ELEMS * sizeof(int16_t),
                       cudaMemcpyDeviceToHost));

  for (size_t ch = 0; ch < NR_CHANNELS; ++ch) {
    for (size_t pkt = 0; pkt < NR_PACKETS_PLUS2; ++pkt) {
      for (size_t fpga = 0; fpga < NR_FPGAS; ++fpga) {
        const size_t arrival_idx = (ch * NR_PACKETS_PLUS2 + pkt) * NR_FPGAS + fpga;
        const bool should_be_zero = missing[arrival_idx];
        for (size_t recv_in_pkt = 0; recv_in_pkt < NR_RECEIVERS_PER_PACKET; ++recv_in_pkt) {
          const size_t receiver = fpga * NR_RECEIVERS_PER_PACKET + recv_in_pkt;
          for (size_t pol = 0; pol < NR_POLARIZATIONS; ++pol) {
            const size_t idx =
                ((ch * NR_PACKETS_PLUS2 + pkt) * NR_RECEIVERS + receiver) * NR_POLARIZATIONS + pol;
            if (should_be_zero) {
              EXPECT_EQ(result[idx], 0) << "ch=" << ch << " pkt=" << pkt << " fpga=" << fpga;
            } else {
              EXPECT_EQ(result[idx], host_scales[idx]) << "ch=" << ch << " pkt=" << pkt
                                                       << " fpga=" << fpga << " (should be untouched)";
            }
          }
        }
      }
    }
  }

  cudaFree(scales_dev);
  cudaFreeHost(arrivals_host);
}
