#pragma once

// This raw-packet capture backend requires libibverbs (RDMA). The entire file
// is compiled only when CMake found libibverbs and defined HAVE_IBVERBS;
// otherwise it expands to nothing, so builds on machines without an RDMA stack
// stay green and apps that include it simply see no LibibverbsPacketCapture
// symbol.
#ifdef HAVE_IBVERBS

#include <algorithm>
#include <arpa/inet.h>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <getopt.h>
#include <ifaddrs.h>
#include <immintrin.h>
#include <infiniband/verbs.h>
#include <iostream>
#include <netinet/in.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include <netinet/ether.h>
#include <netinet/ip.h>
#include <netinet/udp.h>

#include <pcap.h>
#include <sys/time.h> // Include for gettimeofday()

#include "spatial/gpudirect_relocate.cuh"
#include "spatial/logging.hpp"
#include "spatial/spatial.hpp"

// ---------------------------------------------------------------------------
// Shared NIC/QP/flow-steering setup, used by both LibibverbsPacketCapture
// (CPU-memory ingest) and LibibverbsGpuDirectPacketCapture (GPUDirect RDMA
// ingest, see /home/ubuntu/.claude/plans/i-want-to-start-breezy-lampson.md).
// Factored out to file scope (rather than private methods on one class) so
// neither class needs to inherit from the other -- their receive-path memory
// layouts are different enough (single CPU buffer vs. GPU-resident landing
// pool + CPU header ring) that composing via a shared constructor doesn't fit
// cleanly, but the NIC-facing setup is identical.
// ---------------------------------------------------------------------------
namespace libibverbs_detail {

inline void ibname_to_ethname(const char *ibname, char *ethname) {
  char path[256];
  snprintf(path, sizeof(path), "/sys/class/infiniband/%s/device/net", ibname);

  DIR *dir = opendir(path);
  if (!dir) {
    perror("opendir");
    return;
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 &&
        strcmp(entry->d_name, "..") != 0) {
      printf("  %s\n", entry->d_name);
      strcpy(ethname, entry->d_name);
      closedir(dir);
      return;
    }
  }
  closedir(dir);
}

inline void get_interface_ip(const char *interface_name, struct sockaddr_in *addr) {
  struct ifaddrs *ifaddr;
  if (getifaddrs(&ifaddr) == -1) {
    std::cerr << "Failed to get network interfaces: " << strerror(errno)
              << " (errno: " << errno << ")" << std::endl;
    return;
  }

  for (struct ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr || strcmp(ifa->ifa_name, interface_name) != 0) {
      continue;
    }
    if (ifa->ifa_addr->sa_family == AF_INET) {
      char ip[INET_ADDRSTRLEN];
      *addr = *(struct sockaddr_in *)ifa->ifa_addr;
      inet_ntop(AF_INET, &addr->sin_addr, ip, INET_ADDRSTRLEN);
      freeifaddrs(ifaddr);
      return;
    }
  }

  std::cerr << "No IPv4 address found for interface: " << interface_name << std::endl;
  freeifaddrs(ifaddr);
  addr = nullptr;
}

inline struct ibv_flow *create_udp_flow(ibv_qp *qp, uint16_t udp_dport,
                                        uint32_t src_ip = 0) {
  size_t flow_size = sizeof(struct ibv_flow_attr) + sizeof(struct ibv_flow_spec_eth) +
                     sizeof(struct ibv_flow_spec_ipv4) +
                     sizeof(struct ibv_flow_spec_tcp_udp);
  void *flow_mem = calloc(1, flow_size);

  struct ibv_flow_attr *flow_attr = (struct ibv_flow_attr *)flow_mem;
  struct ibv_flow_spec_eth *eth_spec = (struct ibv_flow_spec_eth *)(flow_attr + 1);
  struct ibv_flow_spec_ipv4 *ipv4_spec = (struct ibv_flow_spec_ipv4 *)(eth_spec + 1);
  struct ibv_flow_spec_tcp_udp *udp_spec =
      (struct ibv_flow_spec_tcp_udp *)(ipv4_spec + 1);

  flow_attr->type = IBV_FLOW_ATTR_NORMAL;
  flow_attr->size = flow_size;
  flow_attr->priority = 0;
  flow_attr->num_of_specs = 3; // ETH + IPv4 + UDP
  flow_attr->port = 1;
  flow_attr->flags = 0;

  eth_spec->type = IBV_FLOW_SPEC_ETH;
  eth_spec->size = sizeof(struct ibv_flow_spec_eth);
  memset(&eth_spec->val, 0, sizeof(eth_spec->val));
  memset(&eth_spec->mask, 0, sizeof(eth_spec->mask));

  ipv4_spec->type = IBV_FLOW_SPEC_IPV4;
  ipv4_spec->size = sizeof(struct ibv_flow_spec_ipv4);
  memset(&ipv4_spec->val, 0, sizeof(ipv4_spec->val));
  memset(&ipv4_spec->mask, 0, sizeof(ipv4_spec->mask));
  if (src_ip != 0) {
    ipv4_spec->val.src_ip = htonl(src_ip);
    ipv4_spec->mask.src_ip = 0xFFFFFFFF;
    struct in_addr ip_addr;
    ip_addr.s_addr = ntohl(src_ip);
    std::cout << "IP Address: " << inet_ntoa(ip_addr) << std::endl;
  }

  udp_spec->type = IBV_FLOW_SPEC_UDP;
  udp_spec->size = sizeof(struct ibv_flow_spec_tcp_udp);
  udp_spec->val.dst_port = htons(udp_dport);
  udp_spec->mask.dst_port = 0xFFFF;
  udp_spec->val.src_port = 0;
  udp_spec->mask.src_port = 0;

  struct ibv_flow *flow = ibv_create_flow(qp, flow_attr);
  if (!flow) {
    std::cerr << "Failed to create flow: " << strerror(errno) << " (errno: " << errno
              << ")" << std::endl;
    free(flow_mem);
    return nullptr;
  }

  free(flow_mem);
  return flow;
}

inline ibv_qp *init_qp(ibv_context *context, ibv_pd *pd, ibv_cq *cq,
                       uint32_t max_recv_wr, uint32_t max_recv_sge) {
  ibv_qp_init_attr qp_init_attr = {};
  qp_init_attr.send_cq = cq;
  qp_init_attr.recv_cq = cq;
  qp_init_attr.cap.max_send_wr = 1;
  qp_init_attr.cap.max_recv_wr = max_recv_wr;
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = max_recv_sge;
  qp_init_attr.qp_type = IBV_QPT_RAW_PACKET;

  ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
  if (!qp) {
    std::cerr << "Failed to create QP: " << strerror(errno) << " (errno: " << errno
              << ")" << std::endl;
    exit(1);
  }

  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = 1;

  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PORT)) {
    std::cerr << "Failed to modify QP to INIT: " << strerror(errno)
              << " (errno: " << errno << ")" << std::endl;
    exit(1);
  }

  attr.qp_state = IBV_QPS_RTR;
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
    std::cerr << "Failed to modify QP to RTR: " << strerror(errno)
              << " (errno: " << errno << ")" << std::endl;
    exit(1);
  }

  attr.qp_state = IBV_QPS_RTS;
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
    std::cerr << "Failed to modify QP to RTS: " << strerror(errno)
              << " (errno: " << errno << ")" << std::endl;
    exit(1);
  }

  return qp;
}

inline void print_device_list() {
  int num_devices = 0;
  ibv_device **device_list = ibv_get_device_list(&num_devices);
  if (!device_list) {
    std::cerr << "Failed to get IB devices list: " << strerror(errno)
              << " (errno: " << errno << ")" << std::endl;
    exit(-1);
  }
  std::cout << "Available devices:" << std::endl;
  for (int i = 0; i < num_devices; ++i) {
    const char *ibname = ibv_get_device_name(device_list[i]);
    char ethname[256] = {};
    ibname_to_ethname(ibname, ethname);
    std::cout << "Device " << i << ": " << ibname << " " << ethname << std::endl;
  }
  ibv_free_device_list(device_list);
}

// Resolves ifname to its backing RDMA device and opens a context/pd/cq/qp
// triple, steered by a UDP-dest-port flow. Shared setup for both capture
// classes below.
struct QpSetup {
  ibv_context *ctx = nullptr;
  ibv_pd *pd = nullptr;
  ibv_cq *cq = nullptr;
  ibv_qp *qp = nullptr;
  ibv_flow *flow = nullptr;
};

// `create_flow`: skip when the QP is send-only (e.g. ibverbs_pcap_sender.cpp)
// -- a UDP-dest-port receive-steering flow is meaningless for a QP that never
// posts receive WRs.
inline QpSetup setup_qp(const std::string &ifname, int port, int num_frames,
                        uint32_t max_recv_sge, bool create_flow = true) {
  print_device_list();

  int num_devices = 0;
  ibv_device **device_list = ibv_get_device_list(&num_devices);
  if (!device_list) {
    throw std::runtime_error("ibv_get_device_list failed: " +
                             std::string(strerror(errno)));
  }
  ibv_device *dev = nullptr;
  for (int i = 0; i < num_devices; ++i) {
    char ethname[256] = {};
    ibname_to_ethname(ibv_get_device_name(device_list[i]), ethname);
    if (ifname == ethname) {
      dev = device_list[i];
      break;
    }
  }
  if (!dev) {
    ibv_free_device_list(device_list);
    throw std::runtime_error("No RDMA device backs interface " + ifname);
  }

  QpSetup s;
  s.ctx = ibv_open_device(dev);
  ibv_free_device_list(device_list);
  if (!s.ctx) throw std::runtime_error("ibv_open_device failed for " + ifname);

  s.pd = ibv_alloc_pd(s.ctx);
  if (!s.pd) throw std::runtime_error("ibv_alloc_pd failed for " + ifname);

  // CQ depth is deliberately larger than the WR count (num_frames): if
  // completions arrive faster than the polling thread drains them -- e.g. a
  // burst of already-arrived packets flooding in the instant the QP goes
  // live, after the multi-second NVRTC startup compile during which nothing
  // was polling yet -- a CQ sized exactly to num_frames can overrun. mlx5's
  // behaviour on CQ overrun is to silently enter a permanent error state
  // and stop generating completions entirely, with no exception/error text
  // from this code (which doesn't check for the async IBV_EVENT_CQ_ERR
  // event) -- previously observed as a hard, deterministic stall at a fixed
  // packet count early in every run, unaffected by fixes to the downstream
  // WR-repost logic.
  s.cq = ibv_create_cq(s.ctx, num_frames * 4, nullptr, nullptr, 0);
  if (!s.cq) throw std::runtime_error("ibv_create_cq failed for " + ifname);

  s.qp = init_qp(s.ctx, s.pd, s.cq, static_cast<uint32_t>(num_frames), max_recv_sge);

  if (create_flow) {
    s.flow = create_udp_flow(s.qp, static_cast<uint16_t>(port));
    if (!s.flow) throw std::runtime_error("create_udp_flow failed for " + ifname);
  }

  return s;
}

inline void teardown_qp(QpSetup &s) {
  if (s.flow) ibv_destroy_flow(s.flow);
  if (s.qp) ibv_destroy_qp(s.qp);
  if (s.cq) ibv_destroy_cq(s.cq);
  if (s.pd) ibv_dealloc_pd(s.pd);
  if (s.ctx) ibv_close_device(s.ctx);
}

} // namespace libibverbs_detail

class LibibverbsPacketCapture : public PacketInput {
private:
  void post_recv(ibv_qp *qp, ibv_mr *header_mr, ibv_mr *mr, char *header_buffer,
                 char *data_buffer, uint32_t frame_id, uint32_t qpid,
                 bool separate_header) {
    ibv_sge sge[2] = {}; // Array to hold up to 2 SGEs

    if (separate_header) {
      // First SGE: Headers in the header buffer
      sge[0].addr = reinterpret_cast<uintptr_t>(header_buffer + frame_id * MTU);
      sge[0].length = TOTAL_HDR_SIZE; // Length of the headers
      sge[0].lkey = header_mr->lkey;

      // Second SGE: Rest of the packet in the data buffer
      sge[1].addr = reinterpret_cast<uintptr_t>(data_buffer + frame_id * MTU);
      sge[1].length = MTU - TOTAL_HDR_SIZE; // Remaining packet size
      sge[1].lkey = mr->lkey;
    } else {
      // Single SGE: Entire packet in the data buffer
      sge[0].addr = reinterpret_cast<uintptr_t>(data_buffer + frame_id * MTU);
      sge[0].length = MTU; // Full packet size
      sge[0].lkey = mr->lkey;
    }

    ibv_recv_wr wr = {};
    wr.wr_id = static_cast<uint64_t>(qpid) << 32 |
               frame_id; // Encode QP ID and frame ID
    wr.sg_list = sge;
    wr.num_sge = separate_header
                     ? 2
                     : 1; // Use 2 SGEs if save_gpu is enabled, otherwise 1

    ibv_recv_wr *bad_wr;
    if (ibv_post_recv(qp, &wr, &bad_wr)) {
      std::cerr << "Failed to post receive work request: " << strerror(errno)
                << " (errno: " << errno << ")" << std::endl;
      exit(1);
    }
  }

public:
  // Construct a raw-packet capture bound to one NIC (`ifname`, e.g.
  // "enp216s0np0"), steering UDP traffic destined to `port` into a dedicated
  // RAW_PACKET QP. One object (and one capture thread) per NIC -- mirrors
  // KernelSocketPacketCapture so the apps' existing per-NIC threading is
  // unchanged. `buffer_size` is accepted only for signature parity with
  // KernelSocketPacketCapture; captured frames always land in MTU-sized slots.
  LibibverbsPacketCapture(std::string &ifname, int port, int buffer_size,
                          int thread_id = 0, int nr_threads = 1)
      : ifname_(ifname), port_(port), thread_id_(thread_id),
        nr_threads_(nr_threads) {
    (void)buffer_size;

    auto s = libibverbs_detail::setup_qp(ifname_, port_, num_frames, /*max_recv_sge=*/2);
    ctx_ = s.ctx;
    pd_ = s.pd;
    cq_ = s.cq;
    qp_ = s.qp;
    flow_ = s.flow;

    // One contiguous, registered CPU buffer; frame i occupies
    // [i*MTU,(i+1)*MTU).
    const size_t buf_bytes = static_cast<size_t>(num_frames) * MTU;
    cpu_buffer_ = static_cast<uint8_t *>(malloc(buf_bytes));
    if (!cpu_buffer_)
      throw std::runtime_error("capture buffer alloc failed for " + ifname_);
    mr_ = ibv_reg_mr(pd_, cpu_buffer_, buf_bytes, IBV_ACCESS_LOCAL_WRITE);
    if (!mr_)
      throw std::runtime_error("ibv_reg_mr failed for " + ifname_);

    INFO_LOG("libibverbs capture ready on {} (udp port {})", ifname_, port_);
  }

  ~LibibverbsPacketCapture() override {
    libibverbs_detail::QpSetup s{ctx_, pd_, cq_, qp_, flow_};
    libibverbs_detail::teardown_qp(s);
    if (ring_mr_)
      ibv_dereg_mr(ring_mr_);
    if (header_ring_zc_mr_)
      ibv_dereg_mr(header_ring_zc_mr_);
    if (mr_)
      ibv_dereg_mr(mr_);
    if (cpu_buffer_)
      free(cpu_buffer_);
  }

  // Register the ProcessorState ring and replace the staging receive buffer
  // with the ring slots themselves. This is called after state construction,
  // because the ring does not exist when the capture object is created.
  void arm_zero_copy(ProcessorStateBase &state) override {
    if (zero_copy_) return;
    void *base = state.packet_ring_base();
    const size_t bytes = state.packet_ring_bytes();
    const size_t stride = state.packet_ring_slot_stride();
    const size_t offset = state.packet_ring_slot_offset();
    if (!base || bytes == 0 || stride == 0 || state.packet_ring_slot_count() == 0)
      throw std::runtime_error("ibverbs zero-copy requires a ProcessorState packet ring");
    if (state.slot_data_capacity() < 64)
      throw std::runtime_error("ibverbs zero-copy packet ring slot is too small");

    ring_mr_ = ibv_reg_mr(pd_, base, bytes, IBV_ACCESS_LOCAL_WRITE);
    if (!ring_mr_)
      throw std::runtime_error("ibv_reg_mr failed for the ProcessorState packet ring");

    direct_slot_bytes_ = state.slot_data_capacity();
    const size_t slots_per_thread =
        state.packet_ring_slot_count() /
        static_cast<size_t>(std::max(1, nr_threads_));
    active_frames_ = static_cast<int>(std::min<size_t>(num_frames, slots_per_thread));
    if (active_frames_ <= 0)
      throw std::runtime_error("packet ring has no slots for this ibverbs capture thread");

    header_ring_zc_.assign(static_cast<size_t>(active_frames_) * kHeaderBytes, 0);
    header_ring_zc_mr_ = ibv_reg_mr(pd_, header_ring_zc_.data(),
                                    header_ring_zc_.size(),
                                    IBV_ACCESS_LOCAL_WRITE);
    if (!header_ring_zc_mr_)
      throw std::runtime_error("ibv_reg_mr failed for the zero-copy header ring");

    frame_slot_indices_.resize(active_frames_);
    frame_slot_ptrs_.resize(active_frames_);
    std::vector<void *> ptrs(active_frames_);
    // Match KernelSocketPacketCapture's strided ownership: capture thread i
    // owns ring slots i, i+N, i+2N, ... . Starting every cursor at zero
    // makes all QPs reserve the same slots and stalls after the initial window.
    next_linear_ = static_cast<uint64_t>(thread_id_);
    const int reserved = state.reserve_write_batch_strided(
        thread_id_, next_linear_, active_frames_, ptrs.data(),
        frame_slot_indices_.data());
    if (reserved != active_frames_) {
      throw std::runtime_error("could not reserve initial ibverbs zero-copy ring slots");
    }
    for (int i = 0; i < active_frames_; ++i) {
      frame_slot_ptrs_[i] = ptrs[i];
      post_direct_recv(i, ptrs[i]);
    }
    zero_copy_ = true;
    INFO_LOG("ibverbs zero-copy ring capture armed on {} ({} frames, {} bytes/slot)",
             ifname_, num_frames, direct_slot_bytes_);
  }

  // Busy-poll the CQ and copy each completed frame into the shared CPU ring
  // buffer, then re-post its receive WR. Completions are polled in batches and
  // producer_mutex is taken once per batch (like KernelSocket's recvmmsg batch)
  // to keep shared-ring contention low at 1M+ pps.
  void get_packets(ProcessorStateBase &state) override {
    INFO_LOG("libibverbs receiver thread started for {}", ifname_);
    // Same opt-in affinity as KernelSocketPacketCapture (SPATIAL_CAPTURE_CPUS,
    // one entry per capture thread). Unlike that backend's blocking poll()+
    // recvmmsg, this thread busy-spins ibv_poll_cq() continuously -- leaving
    // it unpinned means the scheduler can park it on one of this host's
    // NIC-softirq-saturated cores, or let it collide with another capture
    // thread or a worker thread on the same core, causing severe head-of-line
    // blocking that (given fairly consistent thread-creation/scheduling order
    // across runs on this box) can reproduce as a deterministic early stall.
    {
      const int cpu =
          nth_cpu_from_list(std::getenv("SPATIAL_CAPTURE_CPUS"), thread_id_);
      if (cpu >= 0) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(cpu, &cs);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) == 0)
          INFO_LOG("ibverbs capture thread {} pinned to CPU {}", thread_id_, cpu);
        else
          INFO_LOG("ibverbs capture thread {}: failed to pin to CPU {}",
                    thread_id_, cpu);
      }
    }
    if (!zero_copy_) {
      // The constructor cannot post receives yet: zero-copy arming happens
      // after ProcessorState allocates the ring. Keep the legacy staging path
      // identical, but defer its initial WRs until we know which mode is used.
      for (int jframe = 0; jframe < num_frames; ++jframe)
        repost(jframe);
    }
    ibv_wc wc[POLL_BATCH];
    // Diagnostic: safe, no syscalls beyond ibv_poll_cq itself -- just a
    // periodic wall-clock-gated counter dump so we can see whether this
    // thread is actually looping (and how fast) versus genuinely blocked
    // somewhere, without risking a diagnostic that can itself hang (an
    // earlier version called ibv_get_async_event() here, which blocks by
    // default -- removed after it looked like it might be the reason a test
    // run produced zero log output for the entire capture window).
    uint64_t empty_polls = 0;
    uint64_t total_polls = 0;
    uint64_t nonzero_polls = 0;
    auto last_diag = std::chrono::steady_clock::now();

    while (state.running.load(std::memory_order_acquire)) {
      int n = ibv_poll_cq(cq_, POLL_BATCH, wc);
      ++total_polls;
      if (total_polls <= 20) {
        std::cerr << "ITER ibverbs " << ifname_ << " thread " << thread_id_
                  << " iter=" << total_polls << " n=" << n << std::endl;
      }
      const auto now = std::chrono::steady_clock::now();
      if (now - last_diag > std::chrono::seconds(2)) {
        last_diag = now;
        // Bypass spatial::Logger entirely for this one -- straight to
        // stderr, which the shell capture already reliably picks up,
        // in case the shared logger singleton has its own issue.
        std::cerr << "DIAG ibverbs " << ifname_ << " thread " << thread_id_
                  << ": " << total_polls << " polls, " << nonzero_polls
                  << " nonzero, " << empty_polls
                  << " empty, received=" << state.packets_received.load()
                  << std::endl;
      }
      if (n == 0) {
        ++empty_polls;
        continue;
      }
      ++nonzero_polls;
      if (n < 0) {
        ERROR_LOG("ibv_poll_cq failed on {}", ifname_);
        break;
      }

      std::unique_lock<std::mutex> legacy_lock(state.producer_mutex,
                                                std::defer_lock);
      if (!zero_copy_) legacy_lock.lock();
      for (int i = 0; i < n; ++i) {
        const uint32_t jframe = static_cast<uint32_t>(wc[i].wr_id & 0xFFFFFFFF);
        uint8_t *frame = zero_copy_
                             ? static_cast<uint8_t *>(frame_slot_ptrs_[jframe])
                             : cpu_buffer_ + static_cast<size_t>(jframe) * MTU;

        if (wc[i].status != IBV_WC_SUCCESS) {
          ERROR_LOG("ibverbs WC error on {}: {}", ifname_,
                    ibv_wc_status_str(wc[i].status));
          if (zero_copy_) {
            const int ring_index = frame_slot_indices_[jframe];
            state.abandon_write_batch(1, &ring_index);
            // Same "never lose the frame slot" reasoning as the success
            // path below -- retry rather than dropping this WR on a
            // transient ring-full condition.
            void *replacement = nullptr;
            int replacement_index = -1;
            const auto retry_deadline2 =
                std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (state.running.load(std::memory_order_acquire)) {
              if (state.reserve_write_batch_strided(thread_id_, next_linear_,
                                                     1, &replacement,
                                                     &replacement_index) == 1)
                break;
              if (std::chrono::steady_clock::now() > retry_deadline2) {
                std::cerr << "TIMEOUT ibverbs " << ifname_ << " thread "
                          << thread_id_
                          << " stuck reserving replacement slot (WC-error "
                             "path) for 3s"
                          << std::endl;
                break;
              }
              _mm_pause();
            }
            if (replacement != nullptr) {
              frame_slot_indices_[jframe] = replacement_index;
              frame_slot_ptrs_[jframe] = replacement;
              post_direct_recv(jframe, replacement);
            }
          } else {
            repost(jframe); // don't lose the staging slot
          }
          continue;
        }

        // wc.byte_len is the total across ALL posted SGEs. The legacy
        // (non-zero-copy) path posts a single SGE holding the whole raw
        // frame, so len == frame length there; the zero-copy path posts two
        // (header discard + payload-only landing zone), so len includes
        // TOTAL_HDR_SIZE header bytes the ring slot itself never received.
        const int len = static_cast<int>(wc[i].byte_len);
        const int payload_len = zero_copy_ ? (len - TOTAL_HDR_SIZE) : len;

        // Recover the source IP from the captured IP header so
        // OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET demux (fpga id = third octet
        // of 10.0.<id>.x) works exactly as it does with recvmmsg's msg_name
        // on the KernelSocket path. Zero-copy's header landed separately in
        // header_ring_zc_[jframe] (not at `frame`, which is payload-only);
        // the legacy path's single SGE still has the full frame at `frame`.
        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        if (zero_copy_) {
          const uint8_t *hdr = header_ring_zc_.data() +
                               static_cast<size_t>(jframe) * kHeaderBytes;
          const IPHeader *ip =
              reinterpret_cast<const IPHeader *>(hdr + sizeof(EthernetHeader));
          addr.sin_addr.s_addr = ip->src_ip; // already network byte order
        } else if (len >= static_cast<int>(sizeof(EthernetHeader) +
                                           sizeof(IPHeader))) {
          const IPHeader *ip = reinterpret_cast<const IPHeader *>(
              frame + sizeof(EthernetHeader));
          addr.sin_addr.s_addr = ip->src_ip; // already network byte order
        }

        if (zero_copy_) {
          const int ring_index = frame_slot_indices_[jframe];
          const int packet_len = std::min<int>(payload_len, direct_slot_bytes_);
          state.commit_write_batch(1, &ring_index, &packet_len, &addr);
          state.packets_received += 1;

          // The completion was already pulled off the hardware CQ (that's
          // destructive -- it cannot be re-polled), so this WR's frame slot
          // *must* get a fresh receive buffer posted before moving on, or
          // this QP's fixed-size frame pool permanently loses one slot. The
          // shared ring being momentarily full is a brief, self-clearing
          // condition once workers drain it (proven elsewhere: NICDrops=0
          // in steady state) -- spin-retry here rather than abandoning the
          // WR, which previously caused the whole receiver to stall forever
          // once enough transient misses (typically right at startup, before
          // workers are warmed up) exhausted the frame pool to zero.
          void *replacement = nullptr;
          int replacement_index = -1;
          const auto retry_deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds(3);
          bool timed_out = false;
          while (state.running.load(std::memory_order_acquire)) {
            if (state.reserve_write_batch_strided(thread_id_, next_linear_, 1,
                                                   &replacement,
                                                   &replacement_index) == 1)
              break;
            if (std::chrono::steady_clock::now() > retry_deadline) {
              std::cerr << "TIMEOUT ibverbs " << ifname_ << " thread "
                        << thread_id_
                        << " stuck reserving replacement slot for 3s "
                           "(next_linear_="
                        << next_linear_
                        << ", received=" << state.packets_received.load()
                        << ") -- giving up on this WR, frame pool shrinks by 1"
                        << std::endl;
              timed_out = true;
              break;
            }
            _mm_pause();
          }
          if (timed_out) {
            continue; // don't repost; move on to the next completion in the batch
          }
          if (replacement == nullptr) {
            // Shutdown raced us out of the retry loop; nothing left to post.
            continue;
          }
          frame_slot_indices_[jframe] = replacement_index;
          frame_slot_ptrs_[jframe] = replacement;
          post_direct_recv(jframe, replacement);
        } else {
          const int copy_len =
              std::min<int>(len, static_cast<int>(state.slot_data_capacity()));
          std::memcpy(state.get_current_write_pointer(), frame, copy_len);
          state.add_received_packet_metadata(len, addr);
          state.packets_received += 1;
          state.get_next_write_pointer();
          repost(jframe); // hand the slot back to the NIC
        }
      }
    }
    if (zero_copy_) {
      // Outstanding direct receives own uncommitted ring slots. Make them
      // reclaimable before the processor performs its shutdown drain.
      for (int i = 0; i < active_frames_; ++i)
        state.abandon_write_batch(1, &frame_slot_indices_[i]);
    }
    INFO_LOG("libibverbs receiver thread exiting for {}", ifname_);
  }

private:
  // Re-arm a single receive WR for frame slot `jframe` (single SGE, CPU path).
  void repost(int jframe) {
    post_recv(qp_, mr_, mr_, reinterpret_cast<char *>(cpu_buffer_),
              reinterpret_cast<char *>(cpu_buffer_), jframe, /*qpid=*/0,
              /*separate_header=*/false);
  }

  void post_direct_recv(int frame_id, void *slot) {
    // Two SGEs, same split as post_recv's separate_header=true path and
    // LibibverbsGpuDirectPacketCapture: the RAW_PACKET QP delivers the full
    // Ethernet+IP+UDP frame, but every downstream consumer of a ring slot's
    // ->data (packet parsing, copy_nt reorder, the correlator) expects
    // payload-only, matching what KernelSocketPacketCapture's recvmmsg
    // (over a SOCK_DGRAM socket, which the kernel already strips headers
    // from) delivers there. A single SGE landing the whole raw frame at
    // ->data[0] silently shifted every field this path's packets were
    // parsed as by TOTAL_HDR_SIZE bytes.
    ibv_sge sge[2] = {};
    sge[0].addr = reinterpret_cast<uintptr_t>(
        header_ring_zc_.data() + static_cast<size_t>(frame_id) * kHeaderBytes);
    sge[0].length = static_cast<uint32_t>(TOTAL_HDR_SIZE);
    sge[0].lkey = header_ring_zc_mr_->lkey;
    sge[1].addr = reinterpret_cast<uintptr_t>(slot);
    sge[1].length = static_cast<uint32_t>(direct_slot_bytes_);
    sge[1].lkey = ring_mr_->lkey;
    ibv_recv_wr wr = {};
    wr.wr_id = static_cast<uint64_t>(frame_id);
    wr.sg_list = sge;
    wr.num_sge = 2;
    ibv_recv_wr *bad_wr = nullptr;
    if (ibv_post_recv(qp_, &wr, &bad_wr)) {
      std::cerr << "Failed to post direct ibverbs receive: " << strerror(errno)
                << " (errno=" << errno << ")" << std::endl;
      exit(1);
    }
  }

  std::string ifname_;
  int port_ = 0;
  ibv_context *ctx_ = nullptr;
  ibv_pd *pd_ = nullptr;
  ibv_cq *cq_ = nullptr;
  ibv_qp *qp_ = nullptr;
  ibv_flow *flow_ = nullptr;
  ibv_mr *mr_ = nullptr;
  uint8_t *cpu_buffer_ = nullptr;
  ibv_mr *ring_mr_ = nullptr;
  size_t direct_slot_bytes_ = 0;
  // Per-frame landing zone for the Ethernet+IP+UDP header bytes a RAW_PACKET
  // QP delivers ahead of the payload (see TOTAL_HDR_SIZE / post_recv's own
  // separate_header split, and LibibverbsGpuDirectPacketCapture's own
  // header_ring_, which this mirrors). post_direct_recv posts every WR with
  // two SGEs so only the payload lands at the ring slot's data pointer, but
  // the header bytes still get read back (source IP -> FPGA id demux, see
  // get_packets()) so each concurrently-outstanding frame_id needs its OWN
  // slot here -- a single shared buffer races when a poll batch contains
  // more than one completion (whichever's DMA landed last "wins" the
  // buffer, not necessarily the one currently being processed).
  static constexpr int kHeaderBytes = 64; // >= 42 (eth+ip+udp), rounded up
  std::vector<uint8_t> header_ring_zc_;
  ibv_mr *header_ring_zc_mr_ = nullptr;
  bool zero_copy_ = false;
  int active_frames_ = num_frames;
  int thread_id_ = 0;
  int nr_threads_ = 1;
  uint64_t next_linear_ = 0;
  std::vector<int> frame_slot_indices_;
  std::vector<void *> frame_slot_ptrs_;

  static constexpr int MTU = 9216;          // >= max jumbo frame, 64B aligned
  static constexpr int TOTAL_HDR_SIZE = 42; // eth(14)+ip(20)+udp(8)
  static constexpr int num_frames = 1024;   // pre-posted recv WRs per QP
  static constexpr int POLL_BATCH = 64;     // completions polled per batch
};

// ---------------------------------------------------------------------------
// GPUDirect RDMA ingest (see /home/ubuntu/.claude/plans/i-want-to-start-breezy-lampson.md
// and docs/architecture.md's GPUDirect section).
//
// UNVERIFIED IN THIS ENVIRONMENT: this class was written and reviewed but
// never compiled -- this container has no libibverbs-dev / infiniband/verbs.h
// at all (HAVE_IBVERBS can't be defined here, see CLAUDE.md's libnl gotcha),
// so nothing below has been built or run. Treat it as a careful first draft
// to validate on real RDMA+GPUDirect hardware, not as tested code -- see the
// plan's "Open risks" section for the specific unverified pieces
// (ibv_reg_mr against device pointers needs nvidia-peermem; the CUDA graph
// parameter-update mechanism; construction/arm-order sequencing).
//
// Per-FPGA channels arrive interleaved on one QP (the FPGA can't assign a
// distinct UDP port per channel -- see the plan's Context section), so a
// packet's destination can't be predicted before it arrives: every packet
// lands in a generic double-buffered landing pool, then gets relocated by a
// CUDA-graph-replayed scatter kernel once its header reveals where it
// actually belongs. Validated in apps/bench_gpu.cu's --relocation-check
// (32 channels/4 FPGA, real LambdaGPUPipeline as the concurrent workload):
// see the plan's Phase 0 RESULT note.
// gpudirect_relocate_kernel is defined in spatial/gpudirect_relocate.cuh
// (included above), shared with tests/test_gpudirect_scatter.cu.

class LibibverbsGpuDirectPacketCapture : public PacketInput {
public:
  static constexpr int POLL_BATCH = 64;      // packets per landing-pool half
  static constexpr int NR_HALVES = 2;        // ping-ponged double buffer

  LibibverbsGpuDirectPacketCapture(std::string &ifname, int port, int buffer_size)
      : ifname_(ifname), port_(port) {
    (void)buffer_size;
    auto s = libibverbs_detail::setup_qp(ifname_, port_,
                                         /*num_frames=*/POLL_BATCH * NR_HALVES,
                                         /*max_recv_sge=*/3);
    ctx_ = s.ctx;
    pd_ = s.pd;
    cq_ = s.cq;
    qp_ = s.qp;
    flow_ = s.flow;

    // Header ring: tiny (eth+ip+udp+CustomHeader, ~64B/slot), CPU pinned --
    // one slot per landing-pool slot across both halves.
    const size_t nr_slots = static_cast<size_t>(POLL_BATCH) * NR_HALVES;
    CUDA_CHECK(cudaHostAlloc((void **)&header_ring_, nr_slots * TOTAL_HDR_SIZE,
                             cudaHostAllocDefault));
    header_mr_ =
        ibv_reg_mr(pd_, header_ring_, nr_slots * TOTAL_HDR_SIZE, IBV_ACCESS_LOCAL_WRITE);
    if (!header_mr_)
      throw std::runtime_error("ibv_reg_mr (header ring) failed for " + ifname_);

    // Mapped-pinned parallel src/dst pointer arrays the relocation kernel
    // reads directly -- host writes plain pointers each batch, no explicit
    // H2D copy call (same technique as bench_gpudirect_scatter_common.hpp's
    // descriptor buffer).
    CUDA_CHECK(cudaHostAlloc((void **)&src_samples_host_,
                             sizeof(void *) * POLL_BATCH, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **)&dst_samples_host_,
                             sizeof(void *) * POLL_BATCH, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **)&src_scales_host_,
                             sizeof(void *) * POLL_BATCH, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **)&dst_scales_host_,
                             sizeof(void *) * POLL_BATCH, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&src_samples_dev_, src_samples_host_, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&dst_samples_dev_, dst_samples_host_, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&src_scales_dev_, src_scales_host_, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&dst_scales_dev_, dst_scales_host_, 0));

    INFO_LOG("libibverbs GPUDirect capture constructed on {} (udp port {}) -- "
             "call arm() once the pipeline's device buffers exist before "
             "starting get_packets()",
             ifname_, port_);
  }

  ~LibibverbsGpuDirectPacketCapture() override {
    libibverbs_detail::QpSetup s{ctx_, pd_, cq_, qp_, flow_};
    libibverbs_detail::teardown_qp(s);
    if (header_mr_) ibv_dereg_mr(header_mr_);
    if (samples_mr_) ibv_dereg_mr(samples_mr_);
    if (scales_mr_) ibv_dereg_mr(scales_mr_);
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (event_[0]) cudaEventDestroy(event_[0]);
    if (event_[1]) cudaEventDestroy(event_[1]);
    if (header_ring_) cudaFreeHost(header_ring_);
    if (landing_samples_) cudaFree(landing_samples_);
    if (landing_scales_) cudaFree(landing_scales_);
    if (src_samples_host_) cudaFreeHost(src_samples_host_);
    if (dst_samples_host_) cudaFreeHost(dst_samples_host_);
    if (src_scales_host_) cudaFreeHost(src_scales_host_);
    if (dst_scales_host_) cudaFreeHost(dst_scales_host_);
  }

  // Must be called exactly once, after `state`'s pipeline has GPU-resident
  // landing buffers (GPUPipeline::gpu_landing_samples_ptr/gpu_landing_scales_ptr
  // return non-null) -- i.e. after both the capture object and the pipeline
  // exist, unlike LibibverbsPacketCapture's constructor-time setup. Registers
  // the landing pool's device memory with ibverbs (requires nvidia-peermem /
  // GPUDirect RDMA support in the running kernel -- environment prerequisite,
  // not something this code can satisfy), captures the relocation kernel into
  // a CUDA graph, and posts the first round of receive WRs.
  void arm(ProcessorStateBase &state) {
    samples_slot_bytes_ = state.gpu_samples_slot_bytes();
    scales_slot_bytes_ = state.gpu_scales_slot_bytes();
    if (samples_slot_bytes_ == 0 || scales_slot_bytes_ == 0) {
      throw std::runtime_error(
          "LibibverbsGpuDirectPacketCapture::arm: ProcessorState/GPUPipeline "
          "don't report GPU-resident landing buffers (gpu_samples_slot_bytes/"
          "gpu_scales_slot_bytes returned 0) -- this backend needs a "
          "GPUDirect-capable pipeline, not the default CPU-memory ingest path");
    }

    const size_t nr_slots = static_cast<size_t>(POLL_BATCH) * NR_HALVES;
    CUDA_CHECK(cudaMalloc(&landing_samples_, nr_slots * samples_slot_bytes_));
    CUDA_CHECK(cudaMalloc(&landing_scales_, nr_slots * scales_slot_bytes_));

    // NOTE (unverified): ibv_reg_mr against a cudaMalloc'd device pointer only
    // succeeds with GPUDirect RDMA support (nvidia-peermem / nv_peer_mem
    // kernel module loaded, matching BAR1/IOMMU configuration) -- see the
    // plan's "Open risks" #1/#2. On a non-GPUDirect-capable host this call is
    // expected to fail; there is no software fallback here by design, since
    // the entire point of this backend is avoiding the host bounce.
    samples_mr_ = ibv_reg_mr(pd_, landing_samples_, nr_slots * samples_slot_bytes_,
                             IBV_ACCESS_LOCAL_WRITE);
    if (!samples_mr_)
      throw std::runtime_error(
          "ibv_reg_mr (GPU samples landing pool) failed for " + ifname_ +
          " -- likely missing GPUDirect RDMA (nvidia-peermem) support");
    scales_mr_ = ibv_reg_mr(pd_, landing_scales_, nr_slots * scales_slot_bytes_,
                            IBV_ACCESS_LOCAL_WRITE);
    if (!scales_mr_)
      throw std::runtime_error(
          "ibv_reg_mr (GPU scales landing pool) failed for " + ifname_);

    int low_prio = 0, high_prio = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, low_prio));

    const bool captured = capture_relocation_graph();
    if (!captured)
      throw std::runtime_error("Failed to capture GPUDirect relocation graph for " +
                               ifname_);

    CUDA_CHECK(cudaEventCreateWithFlags(&event_[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&event_[1], cudaEventDisableTiming));

    // Pre-post every WR in both halves so the QP can absorb the first
    // (POLL_BATCH * NR_HALVES) packets immediately.
    for (int half = 0; half < NR_HALVES; ++half)
      for (int slot = 0; slot < POLL_BATCH; ++slot)
        post_landing_recv(half, slot);

    armed_ = true;
    INFO_LOG("libibverbs GPUDirect capture armed on {} (samples_slot_bytes={} "
             "scales_slot_bytes={})",
             ifname_, samples_slot_bytes_, scales_slot_bytes_);
  }

  // Busy-poll the CQ, resolve each completed packet's true destination via
  // state.resolve_gpu_slot() (using the header captured in header_ring_),
  // and relocate one half's worth of packets at a time via the pre-captured
  // CUDA graph. Requires arm() to have been called first.
  void get_packets(ProcessorStateBase &state) override {
    if (!armed_) {
      ERROR_LOG("LibibverbsGpuDirectPacketCapture::get_packets called before "
                "arm() on {} -- refusing to start", ifname_);
      return;
    }
    INFO_LOG("libibverbs GPUDirect receiver thread started for {}", ifname_);
    ibv_wc wc[POLL_BATCH];
    int collected[NR_HALVES] = {0, 0};

    while (state.running.load(std::memory_order_acquire)) {
      int n = ibv_poll_cq(cq_, POLL_BATCH, wc);
      if (n == 0) continue;
      if (n < 0) {
        ERROR_LOG("ibv_poll_cq failed on {}", ifname_);
        break;
      }

      for (int i = 0; i < n; ++i) {
        const uint32_t half = static_cast<uint32_t>((wc[i].wr_id >> 32) & 0xFFFFFFFF);
        const uint32_t slot = static_cast<uint32_t>(wc[i].wr_id & 0xFFFFFFFF);
        if (half >= NR_HALVES || slot >= static_cast<uint32_t>(POLL_BATCH)) {
          ERROR_LOG("Unexpected wr_id on {}: half={} slot={}", ifname_, half, slot);
          continue;
        }

        if (wc[i].status != IBV_WC_SUCCESS) {
          ERROR_LOG("ibverbs WC error on {}: {}", ifname_,
                    ibv_wc_status_str(wc[i].status));
          stage_self_copy(half, slot); // don't relocate; drop in place
        } else {
          resolve_and_stage(state, half, slot);
        }
        ++collected[half];

        if (collected[half] == POLL_BATCH) {
          relocate_and_repost(half);
          collected[half] = 0;
        }
      }
    }
    INFO_LOG("libibverbs GPUDirect receiver thread exiting for {}", ifname_);
  }

private:
  // Reads the tiny header this slot's completion delivered (SGE1, CPU pinned
  // header_ring_), asks the processor state where it truly belongs, and
  // stages src/dst pointers for the upcoming batch replay. Falls back to a
  // same-slot no-op copy if the packet is stale/unresolved (locate_packet()
  // said so) -- matching the reactive path's "discard, don't corrupt"
  // behaviour instead of writing wrong-slot data.
  void resolve_and_stage(ProcessorStateBase &state, uint32_t half, uint32_t slot) {
    const uint32_t ring_slot = half * POLL_BATCH + slot;
    const uint8_t *hdr_bytes = header_ring_ + ring_slot * TOTAL_HDR_SIZE;
    const CustomHeader *custom =
        reinterpret_cast<const CustomHeader *>(hdr_bytes + 42); // eth+ip+udp

    void *samples_addr = nullptr;
    void *scales_addr = nullptr;
    const bool resolved = state.resolve_gpu_slot(
        custom->sample_count, custom->fpga_id, custom->freq_channel,
        samples_addr, scales_addr);

    if (!resolved) {
      stage_self_copy(half, slot);
      return;
    }

    src_samples_host_[slot] = landing_samples_slot_ptr(half, slot);
    dst_samples_host_[slot] = samples_addr;
    src_scales_host_[slot] = landing_scales_slot_ptr(half, slot);
    dst_scales_host_[slot] = scales_addr;
  }

  // Harmless no-op relocation (src==dst, both the landing slot itself) for a
  // packet that's stale/unresolved or arrived with a WC error -- keeps every
  // slot's src/dst pointers valid for the batch replay without writing
  // anywhere but the landing pool itself.
  void stage_self_copy(uint32_t half, uint32_t slot) {
    void *samples_slot = landing_samples_slot_ptr(half, slot);
    void *scales_slot = landing_scales_slot_ptr(half, slot);
    src_samples_host_[slot] = samples_slot;
    dst_samples_host_[slot] = samples_slot;
    src_scales_host_[slot] = scales_slot;
    dst_scales_host_[slot] = scales_slot;
  }

  // Gates reuse of `half`'s landing-pool slots on its *previous* relocation
  // graph replay having actually finished reading them (the double-buffer
  // safety check validated in bench_gpu.cu's --relocation-check, see the
  // plan's Phase 0 RESULT note) before replaying the graph again with this
  // batch's staged src/dst pointers, then reposts this half's WRs so the NIC
  // can reuse it two batches from now.
  void relocate_and_repost(uint32_t half) {
    if (half_used_[half]) {
      if (cudaEventQuery(event_[half]) != cudaSuccess) {
        ++stalls_;
        while (cudaEventQuery(event_[half]) != cudaSuccess) {
          // busy-wait -- see relocate_and_repost's doc comment
        }
      }
    }

    // src_samples_host_/dst_samples_host_/etc. were staged per-slot by
    // resolve_and_stage()/stage_self_copy() above -- already in the
    // cudaHostAllocMapped arrays the captured graph reads via
    // src_samples_dev_/dst_samples_dev_/etc., so no explicit copy is needed
    // here before replay.
    CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
    CUDA_CHECK(cudaEventRecord(event_[half], stream_));
    half_used_[half] = true;

    for (int slot = 0; slot < POLL_BATCH; ++slot)
      post_landing_recv(half, slot);
  }

  void *landing_samples_slot_ptr(uint32_t half, uint32_t slot) const {
    const size_t ring_slot = static_cast<size_t>(half) * POLL_BATCH + slot;
    return static_cast<uint8_t *>(landing_samples_) + ring_slot * samples_slot_bytes_;
  }
  void *landing_scales_slot_ptr(uint32_t half, uint32_t slot) const {
    const size_t ring_slot = static_cast<size_t>(half) * POLL_BATCH + slot;
    return static_cast<uint8_t *>(landing_scales_) + ring_slot * scales_slot_bytes_;
  }

  // Captures the per-batch relocation work (samples + scales copy for all
  // POLL_BATCH slots of whichever half is about to be drained) into a CUDA
  // graph so replay avoids a plain kernel launch's ~us-scale overhead at the
  // packet rates this backend targets -- see the plan's Design section and
  // Phase 0 RESULT (validated at 32ch/4-FPGA against the real pipeline). One
  // captured graph serves both halves: which half is being drained is
  // encoded entirely in which src/dst pointers were staged before replay,
  // not in the graph itself.
  bool capture_relocation_graph() {
    if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) !=
        cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    gpudirect_relocate_kernel<<<POLL_BATCH, 256, 0, stream_>>>(
        src_samples_dev_, dst_samples_dev_, static_cast<int>(samples_slot_bytes_));
    gpudirect_relocate_kernel<<<POLL_BATCH, 64, 0, stream_>>>(
        src_scales_dev_, dst_scales_dev_, static_cast<int>(scales_slot_bytes_));
    cudaGraph_t graph = nullptr;
    const cudaError_t end_err = cudaStreamEndCapture(stream_, &graph);
    if (end_err != cudaSuccess || graph == nullptr) {
      if (graph != nullptr) cudaGraphDestroy(graph);
      cudaGetLastError();
      return false;
    }
    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiateWithFlags(&exec, graph, 0) != cudaSuccess || exec == nullptr) {
      cudaGraphDestroy(graph);
      cudaGetLastError();
      return false;
    }
    cudaGraphDestroy(graph);
    graph_exec_ = exec;
    return true;
  }

  // Re-arms a single receive WR for (half, slot): SGE1 -> header ring (CPU
  // pinned), SGE2 -> landing_scales_ slot, SGE3 -> landing_samples_ slot.
  void post_landing_recv(uint32_t half, uint32_t slot) {
    const uint32_t ring_slot = half * POLL_BATCH + slot;
    ibv_sge sge[3] = {};
    sge[0].addr = reinterpret_cast<uintptr_t>(header_ring_ + ring_slot * TOTAL_HDR_SIZE);
    sge[0].length = TOTAL_HDR_SIZE;
    sge[0].lkey = header_mr_->lkey;

    sge[1].addr = reinterpret_cast<uintptr_t>(landing_scales_slot_ptr(half, slot));
    sge[1].length = static_cast<uint32_t>(scales_slot_bytes_);
    sge[1].lkey = scales_mr_->lkey;

    sge[2].addr = reinterpret_cast<uintptr_t>(landing_samples_slot_ptr(half, slot));
    sge[2].length = static_cast<uint32_t>(samples_slot_bytes_);
    sge[2].lkey = samples_mr_->lkey;

    ibv_recv_wr wr = {};
    wr.wr_id = (static_cast<uint64_t>(half) << 32) | slot;
    wr.sg_list = sge;
    wr.num_sge = 3;

    ibv_recv_wr *bad_wr = nullptr;
    if (ibv_post_recv(qp_, &wr, &bad_wr)) {
      std::cerr << "Failed to post GPUDirect receive work request: "
                << strerror(errno) << " (errno: " << errno << ")" << std::endl;
      exit(1);
    }
  }

  std::string ifname_;
  int port_ = 0;
  ibv_context *ctx_ = nullptr;
  ibv_pd *pd_ = nullptr;
  ibv_cq *cq_ = nullptr;
  ibv_qp *qp_ = nullptr;
  ibv_flow *flow_ = nullptr;

  uint8_t *header_ring_ = nullptr;
  ibv_mr *header_mr_ = nullptr;
  void *landing_samples_ = nullptr;
  void *landing_scales_ = nullptr;
  ibv_mr *samples_mr_ = nullptr;
  ibv_mr *scales_mr_ = nullptr;
  size_t samples_slot_bytes_ = 0;
  size_t scales_slot_bytes_ = 0;

  // Per-slot src/dst pointers (arbitrary device addresses resolved via
  // resolve_gpu_slot(), not slot indices) -- cudaHostAllocMapped, host writes
  // plain pointers, the captured graph's kernel reads them via the *_dev_
  // mirror with no explicit copy per batch.
  void **src_samples_host_ = nullptr, **dst_samples_host_ = nullptr;
  void **src_scales_host_ = nullptr, **dst_scales_host_ = nullptr;
  // Non-const so cudaHostGetDevicePointer (which takes void** out-param) can
  // write into them directly; implicitly convert to `void *const *` at the
  // gpudirect_relocate_kernel call sites below.
  void **src_samples_dev_ = nullptr;
  void **dst_samples_dev_ = nullptr;
  void **src_scales_dev_ = nullptr;
  void **dst_scales_dev_ = nullptr;

  cudaStream_t stream_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  cudaEvent_t event_[NR_HALVES] = {nullptr, nullptr};
  bool half_used_[NR_HALVES] = {false, false};
  unsigned long long stalls_ = 0;
  bool armed_ = false;

  // eth+ip+udp+CustomHeader, rounded up to 64B (matches PacketEntry::DATA_CAPACITY's
  // rounding in packet_formats.hpp) so header_ring_ + ring_slot*TOTAL_HDR_SIZE stays
  // aligned for the CustomHeader* reinterpret_cast in resolve_and_stage().
  static constexpr int TOTAL_HDR_SIZE =
      (42 + sizeof(CustomHeader) + 63) & ~static_cast<size_t>(63);
};

#endif // HAVE_IBVERBS
