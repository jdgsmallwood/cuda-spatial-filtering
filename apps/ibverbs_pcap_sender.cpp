// Replays a captured .pcap file's raw frames onto the wire via a libibverbs
// RAW_PACKET send QP, for testing LibibverbsPacketCapture /
// LibibverbsGpuDirectPacketCapture (spatial/libibverbs.hpp) end to end on
// real RDMA hardware -- send via ibverbs, receive via ibverbs, on the same
// machine (loopback cable/switch port, or two ports on the same NIC).
//
// Unlike apps/udp_sender.cpp (a plain SOCK_DGRAM socket -- the kernel builds
// its own Ethernet/IP/UDP headers, so it can't reproduce exactly what a raw
// ibverbs receive QP sees), this posts the pcap's captured bytes verbatim as
// one Ethernet frame per ibv_post_send: RAW_PACKET QPs operate at L2, so the
// caller supplies the complete frame (dst/src MAC, IP, UDP, CustomHeader,
// payload) rather than a UDP payload the kernel wraps.
//
// UNVERIFIED IN THIS ENVIRONMENT: no libibverbs-dev / infiniband/verbs.h here
// (same HAVE_IBVERBS constraint as libibverbs.hpp itself -- see that file's
// header comment and CLAUDE.md's libnl gotcha). Written and reviewed
// carefully but never compiled or run; validate on a real RDMA host before
// trusting it.
#ifdef HAVE_IBVERBS

#include "spatial/libibverbs.hpp"
#include <argparse/argparse.hpp>
#include <chrono>
#include <cstring>
#include <infiniband/verbs.h>
#include <iostream>
#include <pcap.h>
#include <thread>

namespace {

// One registered send buffer, reused for every packet: post, poll for the
// send completion, then it's safe to overwrite for the next packet. A test
// tool doesn't need pipelined sends -- simplicity/correctness over
// throughput here, unlike the receive-side landing pool in libibverbs.hpp.
class IbverbsRawSender {
public:
  IbverbsRawSender(const std::string &ifname) {
    auto s = libibverbs_detail::setup_qp(ifname, /*port=*/0, /*num_frames=*/1,
                                        /*max_recv_sge=*/1, /*create_flow=*/false);
    ctx_ = s.ctx;
    pd_ = s.pd;
    cq_ = s.cq;
    qp_ = s.qp;

    buf_ = static_cast<uint8_t *>(malloc(MTU));
    if (!buf_) throw std::runtime_error("send buffer alloc failed for " + ifname);
    mr_ = ibv_reg_mr(pd_, buf_, MTU, IBV_ACCESS_LOCAL_WRITE);
    if (!mr_) throw std::runtime_error("ibv_reg_mr failed for " + ifname);
  }

  ~IbverbsRawSender() {
    if (mr_) ibv_dereg_mr(mr_);
    if (buf_) free(buf_);
    libibverbs_detail::QpSetup s{ctx_, pd_, cq_, qp_, nullptr};
    libibverbs_detail::teardown_qp(s);
  }

  // Sends one frame verbatim (must already include the Ethernet header --
  // exactly what pcap_next_ex() hands back for a link-layer capture) and
  // blocks until the send completes.
  void send_frame(const uint8_t *frame, int len) {
    if (len <= 0 || len > MTU) {
      std::cerr << "Skipping frame of invalid length " << len << std::endl;
      return;
    }
    std::memcpy(buf_, frame, static_cast<size_t>(len));

    ibv_sge sge{};
    sge.addr = reinterpret_cast<uintptr_t>(buf_);
    sge.length = static_cast<uint32_t>(len);
    sge.lkey = mr_->lkey;

    ibv_send_wr wr{};
    wr.wr_id = 0;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;

    ibv_send_wr *bad_wr = nullptr;
    if (ibv_post_send(qp_, &wr, &bad_wr)) {
      throw std::runtime_error(std::string("ibv_post_send failed: ") + strerror(errno));
    }

    // Blocking poll: this tool sends at most one frame in flight, so there's
    // nothing useful to overlap while waiting.
    ibv_wc wc{};
    int n = 0;
    while ((n = ibv_poll_cq(cq_, 1, &wc)) == 0) {
      // busy-wait for the send completion
    }
    if (n < 0) {
      throw std::runtime_error("ibv_poll_cq failed while waiting for send completion");
    }
    if (wc.status != IBV_WC_SUCCESS) {
      throw std::runtime_error(std::string("Send completion error: ") +
                               ibv_wc_status_str(wc.status));
    }
  }

private:
  ibv_context *ctx_ = nullptr;
  ibv_pd *pd_ = nullptr;
  ibv_cq *cq_ = nullptr;
  ibv_qp *qp_ = nullptr;
  ibv_mr *mr_ = nullptr;
  uint8_t *buf_ = nullptr;

  static constexpr int MTU = 9216; // matches LibibverbsPacketCapture's MTU
};

} // namespace

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("ibverbs_pcap_sender");
  program.add_argument("pcap_filename").help("Path to the .pcap/.pcapng file to replay");
  program.add_argument("-i", "--network-interface")
      .help("Network interface to send on (e.g. enp216s0np0)")
      .required();
  program.add_argument("--loop")
      .help("Replay the file repeatedly instead of stopping at EOF")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--pace-us")
      .help("Microseconds to sleep between packets (0 = send as fast as the "
            "blocking-poll send path allows)")
      .default_value(0)
      .scan<'i', int>();
  program.add_argument("--log-every")
      .help("Log a progress line every N packets sent (0 = never)")
      .default_value(1000)
      .scan<'i', int>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << "\n" << program;
    return 1;
  }

  const std::string pcap_filename = program.get<std::string>("pcap_filename");
  const std::string ifname = program.get<std::string>("--network-interface");
  const bool loop = program.get<bool>("--loop");
  const int pace_us = program.get<int>("--pace-us");
  const int log_every = program.get<int>("--log-every");

  char errbuf[PCAP_ERRBUF_SIZE];
  pcap_t *handle = pcap_open_offline(pcap_filename.c_str(), errbuf);
  if (!handle) {
    std::cerr << "pcap_open_offline failed: " << errbuf << std::endl;
    return 1;
  }

  IbverbsRawSender sender(ifname);
  std::cout << "ibverbs_pcap_sender: replaying " << pcap_filename << " on " << ifname
            << (loop ? " (looping)" : "") << std::endl;

  const auto t0 = std::chrono::steady_clock::now();
  uint64_t packets_sent = 0;
  bool keep_going = true;
  while (keep_going) {
    struct pcap_pkthdr *header;
    const u_char *packet;
    int res;
    while ((res = pcap_next_ex(handle, &header, &packet)) >= 0) {
      if (res == 0) continue; // timeout, shouldn't happen on an offline file

      sender.send_frame(packet, static_cast<int>(header->len));
      ++packets_sent;
      if (log_every > 0 && packets_sent % static_cast<uint64_t>(log_every) == 0) {
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "  sent " << packets_sent << " packets (" << (packets_sent / elapsed)
                  << " pkt/sec)" << std::endl;
      }
      if (pace_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(pace_us));
    }
    if (res == -1) {
      std::cerr << "Error reading pcap: " << pcap_geterr(handle) << std::endl;
    }

    if (loop) {
      pcap_close(handle);
      handle = pcap_open_offline(pcap_filename.c_str(), errbuf);
      if (!handle) {
        std::cerr << "pcap_open_offline (loop restart) failed: " << errbuf << std::endl;
        keep_going = false;
      }
    } else {
      keep_going = false;
    }
  }

  std::cout << "Total packets sent: " << packets_sent << std::endl;
  pcap_close(handle);
  return 0;
}

#else // !HAVE_IBVERBS

#include <iostream>
int main() {
  std::cerr << "ibverbs_pcap_sender was built without HAVE_IBVERBS (no libibverbs found at "
               "configure time) -- rebuild on an RDMA host with libibverbs-dev installed."
            << std::endl;
  return 1;
}

#endif
