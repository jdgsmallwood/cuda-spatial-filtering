// Live receive capacity measurement: production-style two-SGE RAW_PACKET QP,
// either immediate repost or a NUMA-local SPSC ring with a no-op consumer.
#include "spatial/libibverbs.hpp"
#include <argparse/argparse.hpp>
#include <atomic>
#include <array>
#include <chrono>
#include <fstream>
#include <memory>
#include <sstream>
#include <thread>

#ifdef HAVE_IBVERBS
namespace {
constexpr int kFrames = 1024;
constexpr int kRing = 4096;
constexpr int kHeader = 64;
constexpr int kPayload = 2688;
constexpr int kHeaderLanding = 64;
constexpr int kWholeFrame = kPayload + kHeaderLanding;
constexpr int kPoll = 64;

struct alignas(64) Slot {
  std::atomic<uint32_t> state{0}; // free=0, posted=1, completed=2
  alignas(64) uint8_t payload[kWholeFrame];
};

int node_for_nic(const std::string &name) {
  std::ifstream in("/sys/class/net/" + name + "/device/numa_node");
  int node = -1;
  in >> node;
  if (node < 0) throw std::runtime_error("no NUMA node for " + name);
  return node;
}

int node_for_cpu(int cpu) {
  for (int node = 0; node < 8; ++node)
    if (std::ifstream("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                      "/node" + std::to_string(node)).good())
      return node;
  return -1;
}

void pin(int cpu) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpu, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
    throw std::runtime_error("cannot pin to CPU " + std::to_string(cpu));
}

struct Nic {
  std::string name;
  int cpu, worker_cpu, node;
  bool ring_mode, whole_frame;
  std::atomic<bool> running{true};
  std::atomic<uint64_t> received{0}, consumed{0}, errors{0}, wait_spins{0};
  std::atomic<uint64_t> empty_polls{0};
  std::exception_ptr failure;
  std::thread receiver, consumer;
  std::unique_ptr<Slot[]> slots;

  Nic(std::string n, int c, int w, bool ring, bool whole)
      : name(std::move(n)), cpu(c), worker_cpu(w), node(node_for_nic(name)),
        ring_mode(ring), whole_frame(whole) {
    if (node_for_cpu(cpu) != node || node_for_cpu(worker_cpu) != node)
      throw std::runtime_error("CPU/NUMA mismatch for " + name);
  }

  void start() {
    receiver = std::thread([this] {
      try { run(); }
      catch (...) { failure = std::current_exception(); running = false; }
    });
  }
  void stop() {
    running = false;
    if (receiver.joinable()) receiver.join();
    if (consumer.joinable()) consumer.join();
    if (failure) std::rethrow_exception(failure);
  }

  void run() {
    pin(cpu);
    // Allocation and page first-touch occur on the NIC's NUMA node.
    const int count = ring_mode ? kRing : kFrames;
    slots = std::make_unique<Slot[]>(count);
    for (int i = 0; i < count; ++i)
      std::memset(slots[i].payload, 0, kWholeFrame);
    std::array<uint8_t, kFrames * kHeader> headers{};
    auto qp = libibverbs_detail::setup_qp(name, 36001, kFrames, 2);
    ibv_mr *payload_mr = ibv_reg_mr(qp.pd, slots.get(),
        sizeof(Slot) * count, IBV_ACCESS_LOCAL_WRITE);
    ibv_mr *header_mr = ibv_reg_mr(qp.pd, headers.data(),
        headers.size(), IBV_ACCESS_LOCAL_WRITE);
    if (!payload_mr || !header_mr)
      throw std::runtime_error("ibv_reg_mr failed for " + name);
    std::array<int, kFrames> frame_slot{};
    const auto post = [&](int frame, int slot) {
      ibv_sge sges[2]{};
      if (whole_frame) {
        // Start 42 B before an aligned payload. The network header occupies
        // only the first cache line and the existing payload layout survives.
        sges[0].addr = reinterpret_cast<uintptr_t>(slots[slot].payload +
                                                   kHeaderLanding - 42);
        sges[0].length = 42 + kPayload;
        sges[0].lkey = payload_mr->lkey;
      } else {
        sges[0].addr = reinterpret_cast<uintptr_t>(headers.data() + frame*kHeader);
        sges[0].length = 42;
        sges[0].lkey = header_mr->lkey;
        sges[1].addr = reinterpret_cast<uintptr_t>(slots[slot].payload);
        sges[1].length = kPayload;
        sges[1].lkey = payload_mr->lkey;
      }
      ibv_recv_wr wr{};
      wr.wr_id = frame;
      wr.sg_list = sges;
      wr.num_sge = whole_frame ? 1 : 2;
      ibv_recv_wr *bad = nullptr;
      if (ibv_post_recv(qp.qp, &wr, &bad))
        throw std::runtime_error("ibv_post_recv failed for " + name);
      frame_slot[frame] = slot;
    };
    for (int i = 0; i < kFrames; ++i) {
      slots[i].state.store(1, std::memory_order_relaxed);
      post(i, i);
    }
    if (ring_mode) {
      consumer = std::thread([this] {
        try {
          pin(worker_cpu);
          uint64_t cursor = 0;
          while (running.load(std::memory_order_acquire)) {
            Slot &slot = slots[cursor % kRing];
            if (slot.state.load(std::memory_order_acquire) != 2) {
              _mm_pause();
              continue;
            }
            slot.state.store(0, std::memory_order_release);
            ++cursor;
            consumed.store(cursor, std::memory_order_relaxed);
          }
        } catch (...) { failure = std::current_exception(); running = false; }
      });
    }
    ibv_wc wc[kPoll];
    uint64_t next_slot = kFrames;
    while (running.load(std::memory_order_acquire)) {
      const int n = ibv_poll_cq(qp.cq, kPoll, wc);
      if (n < 0) throw std::runtime_error("ibv_poll_cq failed for " + name);
      if (!n) { empty_polls.fetch_add(1, std::memory_order_relaxed); continue; }
      int ready[kPoll];
      int ready_count = 0;
      for (int i = 0; i < n; ++i) {
        const int frame = static_cast<int>(wc[i].wr_id);
        if (frame < 0 || frame >= kFrames || wc[i].status != IBV_WC_SUCCESS) {
          errors.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        received.fetch_add(1, std::memory_order_relaxed);
        if (ring_mode) {
          slots[frame_slot[frame]].state.store(2, std::memory_order_release);
          ready[ready_count++] = frame;
        } else {
          post(frame, frame);
        }
      }
      // Retire the full CQ batch before waiting for the consumer. A slot
      // needed by the first replacement may belong to a later completion.
      for (int i = 0; i < ready_count; ++i) {
          const int frame = ready[i];
          Slot &replacement = slots[next_slot % kRing];
          while (replacement.state.load(std::memory_order_acquire) != 0 &&
                 running.load(std::memory_order_acquire)) {
            wait_spins.fetch_add(1, std::memory_order_relaxed);
            _mm_pause();
          }
          if (!running.load(std::memory_order_acquire)) break;
          replacement.state.store(1, std::memory_order_relaxed);
          post(frame, static_cast<int>(next_slot % kRing));
          ++next_slot;
      }
    }
    running = false;
    if (consumer.joinable()) consumer.join();
    libibverbs_detail::stop_qp(qp);
    ibv_dereg_mr(header_mr);
    ibv_dereg_mr(payload_mr);
    libibverbs_detail::teardown_qp(qp);
  }
};

std::vector<std::string> split(const std::string &s) {
  std::stringstream stream(s);
  std::vector<std::string> out;
  for (std::string part; std::getline(stream, part, ',');) out.push_back(part);
  return out;
}
}

int main(int argc, char **argv) {
  argparse::ArgumentParser args("bench_starspin_ibverbs");
  args.add_argument("--mode").required().help("repost or ring");
  args.add_argument("--layout").default_value(std::string("split"))
      .help("split or whole");
  args.add_argument("--interfaces").default_value(
      std::string("enp94s0np0,enp134s0np0,enp175s0np0,enp216s0np0"));
  args.add_argument("--seconds").default_value(15).scan<'i', int>();
  try {
    args.parse_args(argc, argv);
    const auto mode = args.get<std::string>("--mode");
    if (mode != "repost" && mode != "ring") throw std::runtime_error("bad mode");
    const auto layout = args.get<std::string>("--layout");
    if (layout != "split" && layout != "whole") throw std::runtime_error("bad layout");
    const auto names = split(args.get<std::string>("--interfaces"));
    if (names.size() != 4) throw std::runtime_error("four NICs required");
    const int seconds = args.get<int>("--seconds");
    if (seconds < 2 || seconds > 120) throw std::runtime_error("seconds out of range");
    constexpr int capture_cpu[4] = {1, 11, 13, 15};
    constexpr int worker_cpu[4] = {3, 16, 17, 18};
    std::array<std::unique_ptr<Nic>,4> nics;
    for (int i=0;i<4;++i)
      nics[i] = std::make_unique<Nic>(names[i], capture_cpu[i], worker_cpu[i],
                                      mode=="ring", layout=="whole");
    for (auto &nic : nics) nic->start();
    std::this_thread::sleep_for(std::chrono::seconds(2)); // setup/settle
    std::array<uint64_t,4> before{};
    for (int i=0;i<4;++i) before[i]=nics[i]->received.load();
    const auto start=std::chrono::steady_clock::now();
    for (int sec=1;sec<=seconds;++sec) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      if (sec%5==0 || sec==seconds) {
        std::cout << "t=" << sec;
        for (int i=0;i<4;++i)
          std::cout << " " << names[i] << "=" << nics[i]->received.load()-before[i];
        std::cout << std::endl;
      }
    }
    const double elapsed=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-start).count();
    for (auto &nic : nics) nic->stop();
    uint64_t total=0;
    for (int i=0;i<4;++i) {
      const auto count=nics[i]->received.load()-before[i];
      total+=count;
      std::cout << names[i] << " node=" << nics[i]->node
                << " capture_cpu=" << capture_cpu[i]
                << " consumer_cpu=" << worker_cpu[i]
                << " packets=" << count << " pps=" << count/elapsed
                << " consumed=" << nics[i]->consumed.load()
                << " errors=" << nics[i]->errors.load()
                << " wait_spins=" << nics[i]->wait_spins.load() << '\n';
    }
    std::cout << "mode=" << mode << " layout=" << layout << " seconds=" << elapsed
              << " total_packets=" << total << " total_pps=" << total/elapsed
              << std::endl;
    return 0;
  } catch(const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
#else
int main() { std::cerr << "libibverbs not available\n"; return 1; }
#endif
