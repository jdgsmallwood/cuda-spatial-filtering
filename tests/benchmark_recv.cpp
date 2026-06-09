// Standalone loopback benchmark for KernelSocketPacketCapture receive path.
//
// Spawns a sender thread that blasts UDP packets on lo:36001, and a receiver
// thread that calls recvmmsg in a tight loop.  Reports:
//   - packets/sec received
//   - bytes/sec
//   - syscalls/sec
//   - estimated kernel drops via SO_RXQ_OVFL
//
// Run:
//   ./tests/benchmark_recv [--pps N] [--duration S] [--batch B] [--busy-poll]
//
// Useful with perf:
//   perf stat -e cycles,instructions,cache-misses,LLC-load-misses \
//       ./tests/benchmark_recv --duration 10

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <linux/errqueue.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

static constexpr int PORT = 36001;
static constexpr int PACKET_BYTES = 2622; // typical lambda packet UDP payload
static constexpr int BUFFER_SIZE = 4096;

// ---- helpers ----------------------------------------------------------------

static int make_udp_socket(int port) {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0)
        throw std::runtime_error("socket");
    int reuse = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (bind(fd, (sockaddr *)&addr, sizeof(addr)) < 0)
        throw std::runtime_error("bind");
    return fd;
}

// ---- sender -----------------------------------------------------------------

static void sender_thread(std::atomic<bool> &stop, int target_pps) {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return;

    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    dst.sin_port = htons(PORT);

    uint8_t pkt[PACKET_BYTES];
    std::memset(pkt, 0xAB, PACKET_BYTES);

    const int64_t interval_ns = (target_pps > 0)
        ? (1'000'000'000LL / target_pps)
        : 0;

    auto next = std::chrono::steady_clock::now();
    while (!stop.load(std::memory_order_relaxed)) {
        sendto(fd, pkt, PACKET_BYTES, 0, (sockaddr *)&dst, sizeof(dst));
        if (interval_ns > 0) {
            next += std::chrono::nanoseconds(interval_ns);
            std::this_thread::sleep_until(next);
        }
    }
    close(fd);
}

// ---- receivers: three variants ----------------------------------------------

struct Stats {
    uint64_t packets = 0;
    uint64_t bytes = 0;
    uint64_t syscalls = 0;
    uint64_t drops = 0;  // SO_RXQ_OVFL
};

// --- Variant A: current code (staging buffers + memcpy) ---
static Stats recv_variant_a(int duration_s, int batch_size, bool busy_poll) {
    int fd = make_udp_socket(PORT);

    // SO_RCVBUF: try large value (kernel caps at rmem_max; run sysctl to raise)
    int rcvbuf = 128 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    // SO_RXQ_OVFL: gives us a counter of kernel-dropped packets
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_RXQ_OVFL, &one, sizeof(one));

    if (busy_poll) {
        int poll_us = 50;
        setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &poll_us, sizeof(poll_us));
    }

    struct timeval tv { 1, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // staging buffers — current approach (heap vector)
    std::vector<std::vector<uint8_t>> staging(batch_size,
                                               std::vector<uint8_t>(BUFFER_SIZE));
    std::vector<mmsghdr> msgs(batch_size);
    std::vector<iovec> iovs(batch_size);
    std::vector<sockaddr_in> addrs(batch_size);
    // cmsg space for SO_RXQ_OVFL
    std::vector<std::vector<uint8_t>> cmsgbufs(batch_size,
                                                std::vector<uint8_t>(CMSG_SPACE(sizeof(uint32_t))));

    for (int i = 0; i < batch_size; ++i) {
        iovs[i].iov_base = staging[i].data();
        iovs[i].iov_len = BUFFER_SIZE;
        std::memset(&msgs[i], 0, sizeof(msgs[i]));
        msgs[i].msg_hdr.msg_iov = &iovs[i];
        msgs[i].msg_hdr.msg_iovlen = 1;
        msgs[i].msg_hdr.msg_name = &addrs[i];
        msgs[i].msg_hdr.msg_namelen = sizeof(addrs[i]);
        msgs[i].msg_hdr.msg_control = cmsgbufs[i].data();
        msgs[i].msg_hdr.msg_controllen = cmsgbufs[i].size();
    }

    // "ring" buffer — simulate ring slot copy
    alignas(64) static uint8_t ring[50000][BUFFER_SIZE];
    int ring_idx = 0;

    Stats s{};
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(duration_s);

    while (std::chrono::steady_clock::now() < end) {
        int n = recvmmsg(fd, msgs.data(), batch_size, MSG_WAITFORONE, nullptr);
        if (n <= 0) continue;
        ++s.syscalls;
        for (int i = 0; i < n; ++i) {
            int len = msgs[i].msg_len;
            // Extract SO_RXQ_OVFL drop counter
            for (auto *cmsg = CMSG_FIRSTHDR(&msgs[i].msg_hdr); cmsg;
                 cmsg = CMSG_NXTHDR(&msgs[i].msg_hdr, cmsg)) {
                if (cmsg->cmsg_level == SOL_SOCKET &&
                    cmsg->cmsg_type == SO_RXQ_OVFL) {
                    uint32_t drop;
                    std::memcpy(&drop, CMSG_DATA(cmsg), sizeof(drop));
                    if (drop > s.drops) s.drops = drop;
                }
            }
            // Simulate ring-slot memcpy (the double-copy)
            std::memcpy(ring[ring_idx % 50000], staging[i].data(), len);
            ring_idx++;
            s.bytes += len;
            s.packets++;
            msgs[i].msg_len = 0;
            msgs[i].msg_hdr.msg_controllen = cmsgbufs[i].size();
        }
    }

    close(fd);
    return s;
}

// --- Variant B: flat aligned staging array, no per-packet re-zero ---
static Stats recv_variant_b(int duration_s, int batch_size, bool busy_poll) {
    int fd = make_udp_socket(PORT);

    int rcvbuf = 128 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_RXQ_OVFL, &one, sizeof(one));
    if (busy_poll) {
        int poll_us = 50;
        setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &poll_us, sizeof(poll_us));
    }
    struct timeval tv { 1, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // Flat aligned staging — one allocation, contiguous, cache-friendly
    const int max_batch = 256;
    alignas(64) static uint8_t staging_b[256][BUFFER_SIZE];
    static mmsghdr msgs_b[256];
    static iovec iovs_b[256];
    static sockaddr_in addrs_b[256];
    alignas(64) static uint8_t cmsgbuf_b[256][CMSG_SPACE(sizeof(uint32_t))];

    for (int i = 0; i < max_batch; ++i) {
        iovs_b[i].iov_base = staging_b[i];
        iovs_b[i].iov_len = BUFFER_SIZE;
        std::memset(&msgs_b[i], 0, sizeof(msgs_b[i]));
        msgs_b[i].msg_hdr.msg_iov = &iovs_b[i];
        msgs_b[i].msg_hdr.msg_iovlen = 1;
        msgs_b[i].msg_hdr.msg_name = &addrs_b[i];
        msgs_b[i].msg_hdr.msg_namelen = sizeof(addrs_b[i]);
        msgs_b[i].msg_hdr.msg_control = cmsgbuf_b[i];
        msgs_b[i].msg_hdr.msg_controllen = sizeof(cmsgbuf_b[i]);
    }

    alignas(64) static uint8_t ring_b[50000][BUFFER_SIZE];
    int ring_idx = 0;

    Stats s{};
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(duration_s);
    while (std::chrono::steady_clock::now() < end) {
        int n = recvmmsg(fd, msgs_b, batch_size, MSG_WAITFORONE, nullptr);
        if (n <= 0) continue;
        ++s.syscalls;
        for (int i = 0; i < n; ++i) {
            int len = msgs_b[i].msg_len;
            for (auto *cmsg = CMSG_FIRSTHDR(&msgs_b[i].msg_hdr); cmsg;
                 cmsg = CMSG_NXTHDR(&msgs_b[i].msg_hdr, cmsg)) {
                if (cmsg->cmsg_level == SOL_SOCKET &&
                    cmsg->cmsg_type == SO_RXQ_OVFL) {
                    uint32_t drop;
                    std::memcpy(&drop, CMSG_DATA(cmsg), sizeof(drop));
                    if (drop > s.drops) s.drops = drop;
                }
            }
            std::memcpy(ring_b[ring_idx % 50000], staging_b[i], len);
            ring_idx++;
            s.bytes += len;
            s.packets++;
            msgs_b[i].msg_hdr.msg_controllen = sizeof(cmsgbuf_b[i]);
        }
    }

    close(fd);
    return s;
}

// --- Variant C: direct-to-ring (no intermediate staging copy) ---
// iovecs point directly at ring slots; recvmmsg writes there immediately.
static Stats recv_variant_c(int duration_s, int batch_size, bool busy_poll) {
    int fd = make_udp_socket(PORT);

    int rcvbuf = 128 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_RXQ_OVFL, &one, sizeof(one));
    if (busy_poll) {
        int poll_us = 50;
        setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &poll_us, sizeof(poll_us));
    }
    struct timeval tv { 1, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    const int max_batch = 256;
    alignas(64) static uint8_t ring_c[50000][BUFFER_SIZE];
    static mmsghdr msgs_c[256];
    static iovec iovs_c[256];
    static sockaddr_in addrs_c[256];
    alignas(64) static uint8_t cmsgbuf_c[256][CMSG_SPACE(sizeof(uint32_t))];

    int ring_base = 0; // next ring slot to use

    auto setup_batch = [&](int base) {
        for (int i = 0; i < max_batch; ++i) {
            iovs_c[i].iov_base = ring_c[(base + i) % 50000];
            iovs_c[i].iov_len = BUFFER_SIZE;
            msgs_c[i].msg_hdr.msg_iov = &iovs_c[i];
            msgs_c[i].msg_hdr.msg_iovlen = 1;
            msgs_c[i].msg_hdr.msg_name = &addrs_c[i];
            msgs_c[i].msg_hdr.msg_namelen = sizeof(addrs_c[i]);
            msgs_c[i].msg_hdr.msg_control = cmsgbuf_c[i];
            msgs_c[i].msg_hdr.msg_controllen = sizeof(cmsgbuf_c[i]);
        }
    };
    setup_batch(ring_base);

    Stats s{};
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(duration_s);
    while (std::chrono::steady_clock::now() < end) {
        int n = recvmmsg(fd, msgs_c, batch_size, MSG_WAITFORONE, nullptr);
        if (n <= 0) continue;
        ++s.syscalls;
        for (int i = 0; i < n; ++i) {
            for (auto *cmsg = CMSG_FIRSTHDR(&msgs_c[i].msg_hdr); cmsg;
                 cmsg = CMSG_NXTHDR(&msgs_c[i].msg_hdr, cmsg)) {
                if (cmsg->cmsg_level == SOL_SOCKET &&
                    cmsg->cmsg_type == SO_RXQ_OVFL) {
                    uint32_t drop;
                    std::memcpy(&drop, CMSG_DATA(cmsg), sizeof(drop));
                    if (drop > s.drops) s.drops = drop;
                }
            }
            s.bytes += msgs_c[i].msg_len;
            s.packets++;
            msgs_c[i].msg_hdr.msg_controllen = sizeof(cmsgbuf_c[i]);
        }
        // Re-point iovecs to next ring slots (no memcpy needed)
        ring_base = (ring_base + n) % 50000;
        setup_batch(ring_base);
    }

    close(fd);
    return s;
}

// ---- main -------------------------------------------------------------------

static void print_stats(const char *label, const Stats &s, int duration_s) {
    double mpps = s.packets / 1e6 / duration_s;
    double gbps = s.bytes * 8.0 / 1e9 / duration_s;
    double ksyscalls = s.syscalls / 1e3 / duration_s;
    double avg_batch = s.syscalls ? (double)s.packets / s.syscalls : 0;
    std::cout << label << ":\n"
              << "  packets/sec : " << (s.packets / duration_s) << " ("
              << mpps << " Mpps)\n"
              << "  throughput  : " << gbps << " Gbps\n"
              << "  syscalls/sec: " << ksyscalls << " Ksyscalls/s\n"
              << "  avg batch   : " << avg_batch << " pkts\n"
              << "  kern drops  : " << s.drops << "\n\n";
}

int main(int argc, char **argv) {
    int target_pps = 0;   // 0 = max rate per sender
    int duration_s = 5;
    int batch_size = 64;
    bool busy_poll = false;
    int num_senders = 4;  // multiple senders to saturate the receiver

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--pps") && i+1 < argc) target_pps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--duration") && i+1 < argc) duration_s = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i+1 < argc) batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--senders") && i+1 < argc) num_senders = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--busy-poll")) busy_poll = true;
        else if (!strcmp(argv[i], "--help")) {
            std::cout << "usage: benchmark_recv [--pps N] [--duration S] "
                         "[--batch B] [--senders N] [--busy-poll]\n";
            return 0;
        }
    }

    auto run_variant = [&](const char *label, auto fn) {
        std::atomic<bool> stop{false};
        std::vector<std::thread> txs;
        for (int i = 0; i < num_senders; ++i)
            txs.emplace_back(sender_thread, std::ref(stop), target_pps);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        auto s = fn(duration_s, batch_size, busy_poll);
        stop = true;
        for (auto &t : txs) t.join();
        print_stats(label, s, duration_s);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    };

    std::cout << "=== KernelSocket recv benchmark ===\n"
              << "batch=" << batch_size << "  duration=" << duration_s
              << "s  target_pps=" << (target_pps ? target_pps : -1)
              << "  senders=" << num_senders
              << "  busy_poll=" << busy_poll << "\n\n";

    // Variant A — heap vector staging (current code)
    run_variant("A: heap-vector staging + memcpy (current)", recv_variant_a);

    // Variant B — flat aligned staging + memcpy
    run_variant("B: flat-array staging + memcpy", recv_variant_b);

    // Variant C — direct-to-ring (no memcpy)
    {
        std::atomic<bool> stop{false};
        std::vector<std::thread> txs;
        for (int i = 0; i < num_senders; ++i)
            txs.emplace_back(sender_thread, std::ref(stop), target_pps);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        auto s = recv_variant_c(duration_s, batch_size, busy_poll);
        stop = true;
        for (auto &t : txs) t.join();
        print_stats("C: direct-to-ring (no memcpy)", s, duration_s);
    }

    return 0;
}
