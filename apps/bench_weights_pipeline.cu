// Adaptive GPU throughput: pinned input through fine-channel beam formation.
// Excludes packet parsing and output copies/writing; compare with app replay separately.
#include "spatial/output.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/spatial.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

#ifndef NR_OBSERVING_FINE_CHANNEL_LOWER_EXTRA_TRIM
#define NR_OBSERVING_FINE_CHANNEL_LOWER_EXTRA_TRIM 0
#endif
using Config = LambdaConfig<NR_OBSERVING_CHANNELS, 4, 64, 40, 2, 10,
                            256, 1, 40, 32, 56, true, 256, 32, 2, 40,
                            NR_OBSERVING_FINE_CHANNEL_LOWER_EXTRA_TRIM>;

struct BenchState : ProcessorStateBase {
  static constexpr int kBuffers = 8;
  std::atomic<bool> ready[kBuffers];
  BenchState() { for (auto &r : ready) r.store(true); }
  size_t input_buffer_count() const override { return kBuffers; }
  void release_buffer(int i) override { ready[i].store(true, std::memory_order_release); }
  void *get_next_write_pointer() override { return nullptr; }
  void *get_current_write_pointer() override { return nullptr; }
  void add_received_packet_metadata(int, const sockaddr_in &) override {}
  int reserve_write_batch(int, void **, int *) override { return 0; }
  void commit_write_batch(int, const int *, const int *, const sockaddr_in *) override {}
  void set_pipeline(GPUPipeline *) override {}
  void process_all_available_packets() override {}
  void handle_buffer_completion(bool) override {}
};

template <BeamformingMode Mode>
int run(int repeats, int gpu_buffers, bool protected_source) {
  try {
    CUDA_CHECK(cudaFree(0));
    BenchState state;
    BeamWeightsT<Config> weights{};
    auto *flat = reinterpret_cast<std::complex<__half> *>(&weights);
    for (size_t i = 0; i < sizeof(weights) / sizeof(*flat); ++i)
      flat[i] = {__float2half(1.f / Config::NR_RECEIVERS), __float2half(0)};
    BeamSteering<Config> steering({}, {}, {}, FrequencyPlan{}, 0, ArrayLocation{}, 1., gpu_buffers);
    std::unordered_map<int, int> ranks;
    for (int c = 0; c < Config::NR_CHANNELS; ++c)
      ranks[c] = Mode == BeamformingMode::WeightsOnly ? 0 : 2;
    AdaptiveBeamPipeline<Config, Mode> pipeline(
        gpu_buffers, &weights, ranks, 0, std::move(steering), false, false, 3.f, 10., 1.f/64);
    if (protected_source) {
      constexpr size_t channels = Mode == BeamformingMode::Coarse ? Config::NR_FPGA_CHANNELS : Config::NR_CHANNELS;
      pipeline.set_protected_signatures(std::vector<std::complex<float>>(
          channels * Config::NR_POLARIZATIONS * Config::NR_RECEIVERS, {1.f, 0.f}));
    }
    pipeline.set_state(&state);
    std::vector<std::unique_ptr<Config::PacketFinalDataType>> packets;
    for (int i = 0; i < BenchState::kBuffers; ++i) {
      packets.emplace_back(std::make_unique<Config::PacketFinalDataType>());
      packets.back()->buffer_index = i;
      packets.back()->start_seq_id = 0;
      packets.back()->end_seq_id = 0;
      auto &packet = *packets.back();
      uint32_t random = 41 + i;
      auto *samples = static_cast<int8_t *>(packet.get_samples_ptr());
      for (size_t j = 0; j < packet.get_samples_elements_size(); ++j) {
        random = random * 1664525u + 1013904223u;
        samples[j] = int8_t((random >> 24) % 17) - 8;
      }
      std::fill_n(static_cast<int16_t *>(packet.get_scales_ptr()),
                  packet.get_scales_element_size() / sizeof(int16_t), int16_t(1));
      std::memset(packet.get_arrivals_ptr(), 1, packet.get_arrivals_size());
    }
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, packets[0]->samples));
    if (attributes.type != cudaMemoryTypeHost) throw std::runtime_error("input is not CUDA-pinned host memory");
    std::cout << "registered_slab=" << (packets[0]->sample_mapping != nullptr)
              << " alignment_2MiB=" << (reinterpret_cast<uintptr_t>(packets[0]->samples) % (2*1024*1024) == 0) << '\n';
    const uintptr_t first = reinterpret_cast<uintptr_t>(packets[0]->samples);
    const uintptr_t last = first + packets[0]->get_samples_elements_size() + packets[0]->get_scales_element_size();
    std::ifstream smaps("/proc/self/smaps");
    std::string line;
    bool overlaps = false;
    size_t huge_kib = 0;
    while (std::getline(smaps, line)) {
      unsigned long start, end;
      size_t value;
      if (std::sscanf(line.c_str(), "%lx-%lx", &start, &end) == 2)
        overlaps = start < last && end > first;
      else if (overlaps && std::sscanf(line.c_str(), "AnonHugePages: %zu kB", &value) == 1)
        huge_kib += value;
    }
    std::cout << "overlapping_vma_anon_huge_kib=" << huge_kib << '\n';
    pipeline.execute_pipeline(packets[0].get(), true);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto begin = std::chrono::steady_clock::now();
    for (int n = 0; n < repeats; ++n) {
      const int i = n % BenchState::kBuffers;
      while (!state.ready[i].exchange(false, std::memory_order_acq_rel))
        std::this_thread::yield();
      pipeline.execute_pipeline(packets[i].get());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    const size_t bytes = size_t(Config::NR_FPGA_CHANNELS) * Config::NR_FPGA_SOURCES * Config::NR_PACKETS_FOR_CORRELATION * Config::NR_TIME_STEPS_PER_PACKET * Config::NR_RECEIVERS_PER_PACKET * Config::NR_POLARIZATIONS * 2;
    if (pipeline.eigensolver_failures()) throw std::runtime_error("eigensolver failure invalidates run");
    std::cout << "mode=" << (Mode == BeamformingMode::WeightsOnly ? "weights-only" : Mode == BeamformingMode::Coarse ? "coarse" : "fine")
              << " protected=" << protected_source << " channels=" << NR_OBSERVING_CHANNELS
              << " fine_bins=" << Config::NR_EFFECTIVE_FINE_CHANNELS
              << " gpu_buffers=" << gpu_buffers
              << " input_bytes_per_block=" << bytes
              << " blocks=" << repeats << " seconds=" << elapsed
              << " blocks_per_s=" << repeats / elapsed
              << " input_GB_per_s=" << repeats * double(bytes) / elapsed / 1e9
              << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}

int main(int argc, char **argv) {
  const int repeats = argc > 1 ? std::atoi(argv[1]) : 96;
  const int buffers = argc > 2 ? std::atoi(argv[2]) : 1;
  const std::string mode = argc > 3 ? argv[3] : "weights-only";
  const bool protect = argc > 4 && std::string(argv[4]) == "protected";
  if (repeats < 1 || repeats > 1000 || buffers < 1 || buffers > 3 ||
      (mode == "weights-only" && protect)) return 1;
  if (mode == "weights-only") return run<BeamformingMode::WeightsOnly>(repeats, buffers, protect);
  if (mode == "coarse") return run<BeamformingMode::Coarse>(repeats, buffers, protect);
  if (mode == "fine") return run<BeamformingMode::Fine>(repeats, buffers, protect);
  std::cerr << "mode must be weights-only, coarse or fine\n";
  return 1;
}
