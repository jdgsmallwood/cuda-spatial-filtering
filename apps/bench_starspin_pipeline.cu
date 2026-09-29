// Synthetic pinned input through the actual 4-FPGA, 36-receiver Starspin GPU
// pipeline, including the integrated visibility copy back to pinned host memory.
#include "spatial/output.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/spatial.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

using Config = LambdaConfig<NR_OBSERVING_CHANNELS, 4, 64, 40, 2, 10,
                            256, 1, 40, 32, 56, true, 256, 32, 2, 36>;

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

int main(int argc, char **argv) {
  const int repeats = argc > 1 ? std::atoi(argv[1]) : 96;
  if (repeats < 1 || repeats > 1000) return 1;
  try {
    BenchState state;
    LambdaStarspinPipeline<Config> pipeline(3, 56);
    pipeline.set_state(&state);
    auto output = std::make_shared<SingleHostMemoryOutput<Config>>();
    pipeline.set_output(output);
    std::vector<std::unique_ptr<Config::PacketFinalDataType>> packets;
    for (int i = 0; i < BenchState::kBuffers; ++i) {
      packets.emplace_back(std::make_unique<Config::PacketFinalDataType>());
      packets.back()->buffer_index = i;
      packets.back()->start_seq_id = 0;
      packets.back()->end_seq_id = 0;
    }
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
    const size_t bytes = packets[0]->get_samples_elements_size() +
                         packets[0]->get_scales_element_size();
    std::cout << "channels=" << NR_OBSERVING_CHANNELS
              << " input_bytes_per_block=" << bytes
              << " blocks=" << repeats << " seconds=" << elapsed
              << " blocks_per_s=" << repeats / elapsed
              << " input_GB_per_s=" << repeats * double(bytes) / elapsed / 1e9
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
