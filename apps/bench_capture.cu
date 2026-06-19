#include "spatial/spatial.hpp"
#include <argparse/argparse.hpp>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

// Minimal ProcessorStateBase that just counts packets/bytes handed to it by
// a PacketInput, without doing any reassembly/processing. This isolates the
// raw capture throughput (recvmmsg + memcpy into ring slots) from everything
// downstream.
class BenchCaptureState : public ProcessorStateBase {
public:
  static constexpr size_t NR_SLOTS = 4096;

  BenchCaptureState() : slots_(new uint8_t[NR_SLOTS][BUFFER_SIZE]) {}
  ~BenchCaptureState() { delete[] slots_; }

  std::atomic<uint64_t> bytes_received{0};

  void *get_next_write_pointer() override { return slots_[0]; }
  void *get_current_write_pointer() override { return slots_[0]; }
  void add_received_packet_metadata(const int length,
                                    const sockaddr_in &) override {
    bytes_received.fetch_add(static_cast<uint64_t>(length),
                             std::memory_order_relaxed);
  }

  // Called under producer_mutex: hand out the next NR_SLOTS-ring slots,
  // recycling them — nothing ever consumes them, this is purely measuring
  // ingestion rate.
  int reserve_write_batch(int max_n, void **slot_ptrs,
                          int *slot_indices) override {
    for (int i = 0; i < max_n; ++i) {
      const size_t idx = next_slot_++ % NR_SLOTS;
      slot_ptrs[i] = slots_[idx];
      slot_indices[i] = static_cast<int>(idx);
    }
    return max_n;
  }

  void commit_write_batch(int n, const int * /*slot_indices*/,
                          const int *lens,
                          const sockaddr_in * /*addrs*/) override {
    uint64_t total = 0;
    for (int i = 0; i < n; ++i) {
      total += static_cast<uint64_t>(lens[i]);
    }
    bytes_received.fetch_add(total, std::memory_order_relaxed);
  }

  void release_buffer(const int) override {}
  void set_pipeline(GPUPipeline *) override {}
  void process_all_available_packets() override {}
  void handle_buffer_completion(bool = false) override {}

private:
  uint8_t (*slots_)[BUFFER_SIZE];
  size_t next_slot_ = 0;
};

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("bench_capture");
  program.add_argument("-i", "--interfaces")
      .help("Comma-separated network interfaces to capture on (e.g. "
            "enp134s0np0,enp175s0np0,enp216s0np0,enp...). Use 'lo' for "
            "loopback testing with udp_sender.")
      .default_value(std::string("lo"));
  program.add_argument("-p", "--port")
      .help("UDP port to listen on")
      .default_value(36001)
      .scan<'i', int>();
  program.add_argument("--duration")
      .help("Duration in seconds to capture for")
      .default_value(10.0)
      .scan<'g', double>();
  program.add_argument("--recv-buffer-size")
      .help("SO_RCVBUF size in bytes per socket")
      .default_value(64 * 1024 * 1024)
      .scan<'i', int>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  const std::string interfaces_arg = program.get<std::string>("-i");
  const int port = program.get<int>("-p");
  const double duration_s = program.get<double>("--duration");
  const int recv_buffer_size = program.get<int>("--recv-buffer-size");

  std::vector<std::string> ifnames;
  {
    std::stringstream ss(interfaces_arg);
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (!item.empty()) {
        ifnames.push_back(item);
      }
    }
  }
  if (ifnames.empty()) {
    std::cerr << "No interfaces given" << std::endl;
    return 1;
  }

  std::cout << "bench_capture: " << ifnames.size()
            << " interface(s), port=" << port << " duration=" << duration_s
            << "s" << std::endl;

  std::vector<std::unique_ptr<BenchCaptureState>> states;
  std::vector<std::unique_ptr<KernelSocketPacketCapture>> captures;
  for (auto &ifname : ifnames) {
    states.push_back(std::make_unique<BenchCaptureState>());
    captures.push_back(std::make_unique<KernelSocketPacketCapture>(
        ifname, port, BUFFER_SIZE, recv_buffer_size));
  }

  std::vector<std::thread> threads;
  for (size_t i = 0; i < captures.size(); ++i) {
    threads.emplace_back(
        [&captures, &states, i]() { captures[i]->get_packets(*states[i]); });
  }

  const auto start = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::duration<double>(duration_s));
  for (auto &state : states) {
    state->running.store(0, std::memory_order_release);
  }
  for (auto &t : threads) {
    t.join();
  }
  const auto end = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(end - start).count();

  uint64_t total_packets = 0;
  uint64_t total_bytes = 0;
  for (size_t i = 0; i < ifnames.size(); ++i) {
    const uint64_t packets = states[i]->packets_received;
    const uint64_t bytes = states[i]->bytes_received.load();
    total_packets += packets;
    total_bytes += bytes;
    std::cout << "[Capture " << ifnames[i] << "] packets=" << packets
              << " bytes=" << bytes << " elapsed=" << elapsed << "s"
              << " packets/sec=" << packets / elapsed
              << " GB/sec=" << static_cast<double>(bytes) / elapsed / 1e9
              << " kernel_drops=" << captures[i]->kernel_drops.load()
              << std::endl;
  }
  std::cout << "[Capture TOTAL] packets=" << total_packets
            << " bytes=" << total_bytes << " elapsed=" << elapsed << "s"
            << " packets/sec=" << total_packets / elapsed
            << " GB/sec=" << static_cast<double>(total_bytes) / elapsed / 1e9
            << std::endl;

  return 0;
}
