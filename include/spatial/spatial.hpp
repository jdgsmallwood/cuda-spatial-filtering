#pragma once
#include "spatial/logging.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline_base.hpp"
#include <complex>
#include <cuda.h>
#include <iostream>
#include <libtcc/Correlator.h>
#include <netinet/in.h>
// #include <sys/socket.h>
#include <algorithm>
#include <atomic>
#include <barrier>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <mutex>
#include <new>
#include <pcap/pcap.h>
#include <poll.h>
#include <queue>
#include <stdatomic.h>
#include <sys/mman.h>
#include <sys/time.h>

#define MIN_PCAP_HEADER_SIZE 64
#define BUFFER_SIZE 4096

#ifndef NR_OBSERVING_PACKET_WORKER_THREADS
#define NR_OBSERVING_PACKET_WORKER_THREADS 3
#endif

#include <cuda_fp16.h>

// template <typename T>
// void eigendecomposition(float *h_eigenvalues, int n, const std::vector<T>
// *A); template <typename T> void d_eigendecomposition(float *d_eigenvalues,
// const int n,
//                           const int num_channels, const int
//                           num_polarizations, T *d_A, cudaStream_t stream);
void ccglib_mma(__half *A, __half *B, float *C, const int n_row,
                const int n_col, const int batch_size, int n_inner = -1);

void ccglib_mma_opt(__half *A, __half *B, float *C, const int n_row,
                    const int n_col, const int batch_size, int n_inner,
                    const int tile_size_x, const int tile_size_y);
template <size_t NR_CHANNELS, size_t NR_FPGA_SOURCES> struct BufferState {
  bool is_ready;
  std::array<uint64_t, NR_FPGA_SOURCES> start_seq;
  std::array<uint64_t, NR_FPGA_SOURCES> end_seq;
  std::bitset<NR_CHANNELS> is_populated;
};

struct Result {
  int64_t closest;
  int remainder;
};

inline Result nearest_multiple(int64_t x, int64_t k) {
  // compute nearest integer multiple index
  int64_t m =
      static_cast<int64_t>(std::llround(static_cast<long double>(x) / k));

  int64_t closest = m * k;
  int remainder = static_cast<int>(x - closest);

  return {closest, remainder};
}

// Non-temporal copy for write-only destinations (the pinned d_samples landing
// buffers): streaming stores bypass the cache and skip the read-for-ownership
// a normal memcpy incurs, roughly halving memory traffic for data the CPU
// never reads back — the GPU DMAs it out.  Falls back to memcpy when the
// destination or size isn't vector-aligned.  The sfence is required: NT
// stores are weakly ordered and must be globally visible before the release
// store that publishes the packet as processed.
inline void copy_nt(void *__restrict__ dst, const void *__restrict__ src,
                    size_t n) {
#if defined(__AVX512F__)
  // 64-byte NT stores (40 ops for 2560 B vs 160 SSE2 ops) — Zen 4 / Skylake-X+
  if ((reinterpret_cast<uintptr_t>(dst) & 63) == 0 && (n & 63) == 0) {
    char *d = static_cast<char *>(dst);
    const char *s = static_cast<const char *>(src);
    for (size_t i = 0; i < n; i += 64) {
      _mm512_stream_si512(
          reinterpret_cast<__m512i *>(d + i),
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(s + i)));
    }
    _mm_sfence();
    return;
  }
#endif
#if defined(__AVX2__)
  if ((reinterpret_cast<uintptr_t>(dst) & 31) == 0 && (n & 31) == 0) {
    char *d = static_cast<char *>(dst);
    const char *s = static_cast<const char *>(src);
    for (size_t i = 0; i < n; i += 32) {
      _mm256_stream_si256(
          reinterpret_cast<__m256i *>(d + i),
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + i)));
    }
    _mm_sfence();
    return;
  }
#elif defined(__SSE2__) || defined(__x86_64__)
  if ((reinterpret_cast<uintptr_t>(dst) & 15) == 0 && (n & 15) == 0) {
    char *d = static_cast<char *>(dst);
    const char *s = static_cast<const char *>(src);
    for (size_t i = 0; i < n; i += 16) {
      _mm_stream_si128(
          reinterpret_cast<__m128i *>(d + i),
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(s + i)));
    }
    _mm_sfence();
    return;
  }
#endif
  std::memcpy(dst, src, n);
}

// Returns the n-th entry of a comma-separated CPU list ("4,5,6"), or -1 if
// the list is missing/short/garbled.  Used for SPATIAL_WORKER_CPUS.
inline int nth_cpu_from_list(const char *list, int n) {
  if (list == nullptr) {
    return -1;
  }
  const char *p = list;
  for (int idx = 0;; ++idx) {
    char *end = nullptr;
    long val = std::strtol(p, &end, 10);
    if (end == p) {
      return -1;
    }
    if (idx == n) {
      return static_cast<int>(val);
    }
    if (*end != ',') {
      return -1;
    }
    p = end + 1;
  }
}

// forward declaration of GPUPipeline.
class GPUPipeline;

class ProcessorStateBase {
public:
  int current_buffer = 0;
  std::atomic<int> write_index = 0;
  std::atomic<int> read_index = 0;
  std::unordered_map<uint32_t, int> fpga_ids;
  bool synchronous_pipeline = false;
  std::atomic<int> running = 1;
  // Atomic: incremented concurrently by every receiver thread.
  std::atomic<uint64_t> packets_received = 0;
  std::atomic<uint64_t> packets_processed = 0;
  uint64_t packets_missing = 0;
  std::atomic<uint64_t> packets_discarded = 0;
  // Diagnostics for the ring-buffer "reserve" logic: packets deferred to
  // future_packet_queue (waiting for their buffer to roll around), and
  // packets that fell through copy_data_to_input_buffer_if_able without
  // being copied, future-queued, or discarded (would otherwise permanently
  // poison their ring slot as processed=false).
  std::atomic<uint64_t> packets_future_queued = 0;
  std::atomic<uint64_t> packets_stuck_unprocessed = 0;
  uint64_t pipeline_runs_queued = 0;
  // Set on the first packet discarded because its freq_channel falls outside
  // [MIN_FREQ_CHANNEL, MIN_FREQ_CHANNEL + NR_CHANNELS). Used to emit a
  // one-time warning -- this is the most common cause of "everything discarded,
  // buffers never initialized" when --min_freq_channel is wrong.
  std::atomic<bool> channel_discard_warned{false};
  std::mutex producer_mutex;

  // ── Strided multi-producer support ──────────────────────────────────────
  // Set nr_capture_threads > 0 before starting capture threads to activate
  // the lock-free strided path.  Each capture thread i is assigned slots
  // i, i+N, i+2N, ... and claims them via reserve_write_batch_strided()
  // without taking producer_mutex.  The consumer reads the per-thread claim
  // watermarks to know how far to scan instead of using write_index.
  //
  // Legacy mode (nr_capture_threads == 0): reserve_write_batch() under
  // producer_mutex, write_index watermark — unchanged.
  static constexpr int MAX_CAPTURE_THREADS = 8;
  struct alignas(64) ThreadClaim {
    std::atomic<uint64_t> linear{0};
  };
  ThreadClaim per_thread_claim[MAX_CAPTURE_THREADS];
  // Per-thread read progress, published by the consumer so each producer can
  // check its own ring-full condition independently of other threads.
  ThreadClaim per_thread_read_linear[MAX_CAPTURE_THREADS];
  // Monotonically increasing read progress for the legacy (non-strided) path.
  std::atomic<uint64_t> read_linear{0};
  int nr_capture_threads{0};  // 0 = legacy mutex path
  // ────────────────────────────────────────────────────────────────────────

  virtual void *get_next_write_pointer() = 0;
  virtual void *get_current_write_pointer() = 0;
  virtual void add_received_packet_metadata(const int length,
                                            const sockaddr_in &client_addr) = 0;

  // Reserve up to max_n ring slots.  Fills slot_ptrs[0..n-1] with data[]
  // pointers and slot_indices[0..n-1] with ring indices.  Must be called
  // under producer_mutex (brief: index arithmetic only).  Sets committed=false
  // so the consumer waits for commit_write_batch() before processing.
  virtual int reserve_write_batch(int max_n, void **slot_ptrs,
                                  int *slot_indices) = 0;

  // Commit the n slots previously reserved.  May be called WITHOUT
  // producer_mutex — each slot is exclusively owned by this producer from
  // reserve until commit.  Writes metadata then sets committed=true (release)
  // so the consumer sees the filled data[].
  virtual void commit_write_batch(int n, const int *slot_indices,
                                  const int *lens,
                                  const sockaddr_in *addrs) = 0;

  // Return reserved-but-unfilled slots to the ring as empty (length=0,
  // processed=true, committed=true) so neither producers nor consumers wait
  // on them.  Used by receive-into-ring producers that reserve a batch before
  // knowing how many datagrams arrive.
  virtual void abandon_write_batch(int n, const int *slot_indices) {}

  // Lock-free strided slot reservation.  thread_id identifies which thread
  // is calling (0..nr_capture_threads-1).  my_linear is the caller's LOCAL
  // monotonic write counter (starts at thread_id, increments by
  // nr_capture_threads per slot).  Pass by reference so the callee advances
  // it.  Updates per_thread_claim[thread_id] so the consumer can compute the
  // scan watermark.
  //
  // Default: falls back to reserve_write_batch() under producer_mutex — safe
  // for stub implementations (BenchCaptureState, test fakes) that don't
  // implement the full strided path.
  virtual int reserve_write_batch_strided(int /*thread_id*/,
                                          uint64_t & /*my_linear*/,
                                          int max_n, void **slot_ptrs,
                                          int *slot_indices) {
    std::lock_guard<std::mutex> lk(producer_mutex);
    return reserve_write_batch(max_n, slot_ptrs, slot_indices);
  }

  // Usable bytes in one ring slot's data[] — producers that receive directly
  // into slots size their iovecs/copies with this.
  virtual size_t slot_data_capacity() const { return BUFFER_SIZE; }

  // Optional DMA-registration layout hooks. Non-ProcessorState test fakes
  // retain inert defaults; ibverbs uses these to receive directly into the
  // packet ring rather than staging through a second buffer.
  virtual void *packet_ring_base() const { return nullptr; }
  virtual size_t packet_ring_bytes() const { return 0; }
  virtual size_t packet_ring_slot_stride() const { return 0; }
  virtual size_t packet_ring_slot_offset() const { return 0; }
  virtual size_t packet_ring_slot_count() const { return 0; }

  virtual void release_buffer(const int buffer_index) = 0;
  virtual void set_pipeline(GPUPipeline *pipeline) = 0;
  virtual void process_all_available_packets() = 0;

  virtual void handle_buffer_completion(bool force_flush = false) = 0;

  // GPUDirect ingest support (see /home/ubuntu/.claude/plans/i-want-to-start-breezy-lampson.md,
  // libibverbs.hpp's LibibverbsGpuDirectPacketCapture). Resolves where a
  // packet with the given (sample_count, fpga_id, freq_channel) belongs,
  // using the same placement arithmetic as the reactive
  // copy_data_to_input_buffer_if_able path (locate_packet()), and returns the
  // GPU-resident samples/scales addresses to relocate its payload to.
  // Returns false for stale/unresolved packets (caller should not relocate)
  // or when the paired GPUPipeline has no GPU-resident landing buffers.
  // Default false is safe for every existing ProcessorStateBase subclass
  // (BenchCaptureState, FakeProcessorState, test fakes) -- only
  // ProcessorState<T,...> overrides this meaningfully.
  virtual bool resolve_gpu_slot(uint64_t /*sample_count*/, uint32_t /*fpga_id*/,
                                uint16_t /*freq_channel*/,
                                void *& /*out_samples_addr*/,
                                void *& /*out_scales_addr*/) {
    return false;
  }
  // Byte size of one packet's samples/scales slot in the GPU-resident
  // landing buffers above -- queried once by LibibverbsGpuDirectPacketCapture::arm().
  // Default 0 signals "not GPUDirect-capable."
  virtual size_t gpu_samples_slot_bytes() const { return 0; }
  virtual size_t gpu_scales_slot_bytes() const { return 0; }
};
template <typename T, size_t NR_INPUT_BUFFERS = 2,
          size_t RING_BUFFER_SIZE = 1000, int WORKER_COUNT = 3>
class ProcessorState : public ProcessorStateBase {
public:
  struct alignas(64) WorkRange {
    int start;
    int end;
    int stride{1};
  };

  // Experimental A/B switch, set before starting processor threads. Keep the
  // established handoff as default until integrated throughput is validated.
  bool use_worker_mailboxes = false;
  bool profile_worker_dispatch = false;
  uint64_t dispatch_batches = 0;
  uint64_t dispatch_slots = 0;
  uint64_t dispatch_wait_ns = 0;
  uint64_t dispatch_total_ns = 0;

  typename T::PacketFinalDataType *d_samples[NR_INPUT_BUFFERS];
  // Pointer array kept for API compatibility; all slots come from one
  // contiguous pool so d_packet_data[i] = &d_packet_data_pool[i].  The
  // contiguous layout keeps ring slots in L3 as the processor threads stride
  // through them.
  typename T::PacketEntryType *d_packet_data[RING_BUFFER_SIZE];
  typename T::PacketEntryType *d_packet_data_pool = nullptr;
  size_t MIN_FREQ_CHANNEL;
  size_t NR_BETWEEN_SAMPLES;
  size_t NR_PACKETS_FOR_CORRELATION;

  std::array<BufferState<T::NR_FPGA_CHANNELS, T::NR_FPGA_SOURCES>, NR_INPUT_BUFFERS>
      buffers;
  alignas(64) std::atomic<uint64_t>
      latest_packet_received[T::NR_FPGA_CHANNELS][T::NR_FPGA_SOURCES];
  mutable std::mutex buffer_index_mutex;
  GPUPipeline *pipeline_;

  // CAS-loop atomic max: updates *target to max(*target, val).
  static void atomic_max_u64(std::atomic<uint64_t> &target, uint64_t val) {
    uint64_t cur = target.load(std::memory_order_relaxed);
    while (cur < val &&
           !target.compare_exchange_weak(cur, val, std::memory_order_release,
                                         std::memory_order_relaxed)) {
    }
  }

  // Constructor / Destructor
  ProcessorState(size_t nr_packets_for_correlation, size_t nr_between_samples,
                 size_t min_freq_channel,
                 std::array<int64_t, T::NR_FPGA_SOURCES> fpga_delays,
                 std::unordered_map<uint32_t, int> fpga_ids_)
      : NR_PACKETS_FOR_CORRELATION(nr_packets_for_correlation),
        NR_BETWEEN_SAMPLES(nr_between_samples),
        MIN_FREQ_CHANNEL(min_freq_channel), fpga_delays(fpga_delays) {
    this->fpga_ids = fpga_ids_;
    for (auto &row : latest_packet_received)
      for (auto &v : row)
        v.store(0, std::memory_order_relaxed);
    // Flat lookup for the per-packet hot path: FPGA ids are small integers
    // (IP third octet when OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET), so an
    // array indexed by id replaces an unordered_map hash per packet.  Ids
    // outside [0,256) fall back to the map.
    fpga_index_lut.fill(-1);
    for (const auto &[id, idx] : fpga_ids) {
      if (id < fpga_index_lut.size()) {
        fpga_index_lut[id] = static_cast<int16_t>(idx);
      }
    }
    std::fill_n(d_samples, NR_INPUT_BUFFERS, nullptr);
    std::fill_n(d_packet_data, RING_BUFFER_SIZE, nullptr);
    for (auto &f : modified_since_last_completion_check)
      f.store(false, std::memory_order_relaxed);
    try {
      // Single allocation for all ring slots keeps them contiguous in memory.
      // The individual d_packet_data[i] pointers still work for all existing
      // call sites, but the hardware prefetcher can now stride through the
      // ring.  2 MB-aligned + MADV_HUGEPAGE so the multi-hundred-MB pool is
      // backed by huge pages, cutting dTLB misses while striding the ring.
      const size_t pool_bytes =
          sizeof(typename T::PacketEntryType) * RING_BUFFER_SIZE;
      void *raw = nullptr;
      if (posix_memalign(&raw, 2 * 1024 * 1024, pool_bytes) != 0) {
        throw std::bad_alloc();
      }
#ifdef MADV_HUGEPAGE
      madvise(raw, pool_bytes, MADV_HUGEPAGE);
#endif
      d_packet_data_pool = static_cast<typename T::PacketEntryType *>(raw);
      for (auto i = 0; i < RING_BUFFER_SIZE; ++i) {
        new (&d_packet_data_pool[i]) typename T::PacketEntryType();
        d_packet_data[i] = &d_packet_data_pool[i];
        d_packet_data[i]->processed.store(true, std::memory_order_relaxed);
        d_packet_data[i]->committed.store(false, std::memory_order_relaxed);
      }

      for (auto i = 0; i < NR_INPUT_BUFFERS; ++i) {
        d_samples[i] = new typename T::PacketFinalDataType();
        d_samples[i]->buffer_index = i;
        if (!d_samples[i])
          throw std::bad_alloc();
      }
    } catch (...) {
      cleanup();
      throw;
    }

    for (auto i = 0; i < T::NR_FPGA_SOURCES; ++i) {
      auto r = nearest_multiple(fpga_delays[i],
                                static_cast<int64_t>(nr_between_samples));
      fpga_delays_packet_aligned[i] = r.closest;
      fpga_delays_subpacket[i] = r.remainder;
    }
  };
  ~ProcessorState() { cleanup(); };
  ProcessorState(const ProcessorState &) = delete;
  ProcessorState &operator=(const ProcessorState &) = delete;

  ProcessorState(ProcessorState &&) = delete;
  ProcessorState &operator=(ProcessorState &&) = delete;
  void set_pipeline(GPUPipeline *pipeline) {
    pipeline_ = pipeline;
    pipeline->set_subpacket_delays(fpga_delays_subpacket.data());
  };
  bool get_next_write_index() {
    int next_write_index = -1;
    bool first_loop = true;
    while (next_write_index < 0 ||
           !d_packet_data[next_write_index]->processed.load(
               std::memory_order_relaxed)) {
      if (first_loop) {
        next_write_index = (write_index.load(std::memory_order_relaxed) + 1) %
                           RING_BUFFER_SIZE;
        first_loop = false;
      } else {
        next_write_index = (next_write_index + 1) % RING_BUFFER_SIZE;
      }
      if (next_write_index == read_index.load(std::memory_order_acquire)) {

        log_ring_full_diagnostics();

        return false;
      }
    }
    write_index.store(next_write_index, std::memory_order_release);
    //  DEBUG_LOG("Next write index is...{}", next_write_index);
    return true;
  };

  // Rate-limited (at most once/sec) snapshot of why the ring buffer is full,
  // logged from get_next_write_index() right before it starts spinning. If
  // the spin never recovers, this is the last thing that gets written and
  // tells you whether packets are backing up in future_packet_queue (a
  // per-FPGA stream not advancing) vs. simply being produced faster than
  // process_packets() can drain them.
  void log_ring_full_diagnostics() {
    static std::chrono::steady_clock::time_point last_log{};
    auto now = std::chrono::steady_clock::now();
    if (now - last_log < std::chrono::seconds(1)) {
      return;
    }
    last_log = now;

    size_t unprocessed_slots = 0;
    for (auto i = 0; i < RING_BUFFER_SIZE; ++i) {
      if (!d_packet_data[i]->processed) {
        ++unprocessed_slots;
      }
    }

    std::array<size_t, T::NR_FPGA_SOURCES> future_queue_sizes{};
    {
      std::lock_guard<std::mutex> lock(future_packet_queue_mutex);
      for (auto i = 0; i < T::NR_FPGA_SOURCES; ++i) {
        future_queue_sizes[i] = future_packet_queue[i].size();
      }
    }

    INFO_LOG("Ring buffer is full!! Dropping packets... "
             "read_index={} write_index={} unprocessed_slots={}/{} "
             "received={} processed={} discarded={} future_queued={} "
             "stuck_unprocessed={}",
             read_index.load(std::memory_order_relaxed),
             write_index.load(std::memory_order_relaxed), unprocessed_slots,
             RING_BUFFER_SIZE, packets_received.load(),
             packets_processed.load(), packets_discarded.load(),
             packets_future_queued.load(), packets_stuck_unprocessed.load());
    for (auto i = 0; i < T::NR_FPGA_SOURCES; ++i) {
      INFO_LOG("  future_packet_queue[fpga_index={}].size() = {}", i,
               future_queue_sizes[i]);
    }
  }

  // Core placement formula for one candidate buffer window: which packet
  // slot (relative to that buffer's start_seq[fpga_index]) this sample_count
  // maps to. Single source of truth -- copy_data_to_input_buffer_if_able's
  // scan loop below and locate_packet() (used by the GPUDirect ingest path,
  // see docs/architecture.md) both call this instead of duplicating the
  // arithmetic.
  int packet_index_for_buffer(uint64_t sample_count, size_t fpga_index,
                              int buffer_index) const {
    const uint64_t buffer_start = buffers[buffer_index].start_seq[fpga_index];
    // sample_count < buffer_start is the *normal* case this function must
    // detect correctly (every packet before its buffer's window opens hits
    // it, and callers rely on a properly negative result to discard/defer
    // it) -- computing the difference in uint64_t underflows and wraps to a
    // huge positive value instead, which the final `int` cast then
    // truncates essentially arbitrarily (sometimes staying visibly out of
    // range, sometimes wrapping back into what looks like a small, "valid"
    // in-buffer index). This was latent for small FPGA delays (tens of
    // millions, per alveo_delays.json) but became a serious problem with a
    // FPGA-restart-sized delay (~-80.8 billion, 2026-09-20 32-channel test):
    // sample_count and buffer_start differ by roughly that much, so nearly
    // every packet from that source hit the underflow. Do the subtraction
    // in signed 64-bit (both operands are always well within int64_t's
    // range, so this cannot itself overflow) and clamp before the final
    // narrowing cast, so a legitimately huge mismatch clamps to an
    // unambiguous "very out of range" sentinel instead of wrapping.
    const int64_t diff = static_cast<int64_t>(sample_count) -
                         static_cast<int64_t>(buffer_start);
    const int64_t index64 = diff / static_cast<int64_t>(NR_BETWEEN_SAMPLES);
    constexpr int64_t kClampBound = 1'000'000'000; // far beyond any real buffer size
    if (index64 < -kClampBound) return static_cast<int>(-kClampBound);
    if (index64 > kClampBound) return static_cast<int>(kClampBound);
    return static_cast<int>(index64);
  }

  enum class PacketLocationStatus {
    kValid,      // packet_index/buffer_index below are where this packet belongs
    kStale,      // older than the lowest possible start token -- discard
    kUnresolved  // didn't land in any of the NR_INPUT_BUFFERS windows scanned
  };
  struct PacketLocation {
    PacketLocationStatus status;
    int buffer_index = -1;
    int packet_index = 0;
  };

  // Resolves the (buffer_index, packet_index) a packet maps to, scanning the
  // NR_INPUT_BUFFERS windows starting at current_buffer exactly as
  // copy_data_to_input_buffer_if_able does. Used by the GPUDirect ingest
  // path's proactive placement (see libibverbs.hpp) instead of duplicating
  // this scan.
  //
  // Known limitation vs. the reactive path below: guard/boundary packets
  // whose packet_index falls in the overlap between two adjacent buffer
  // windows are copied into BOTH buffers by the reactive path (see
  // is_extended/num_copied below) -- this returns only the first (lowest
  // buffer_num) match. Acceptable for the GPUDirect path's first cut since
  // it affects only the ~2 guard packets per buffer rotation, not steady
  // state; revisit if boundary-packet correctness matters for a given
  // consumer.
  PacketLocation locate_packet(uint64_t sample_count, size_t fpga_index) const {
    const int current_buf = current_buffer;
    for (int buffer_num = 0; buffer_num < NR_INPUT_BUFFERS; ++buffer_num) {
      const int buffer_index = (current_buf + buffer_num) % NR_INPUT_BUFFERS;
      const int packet_index =
          packet_index_for_buffer(sample_count, fpga_index, buffer_index);
      if (buffer_num == 0 && packet_index < -1) [[unlikely]] {
        return {PacketLocationStatus::kStale, -1, packet_index};
      }
      if (packet_index >= -1 &&
          packet_index < static_cast<int>(NR_PACKETS_FOR_CORRELATION) + 1) {
        return {PacketLocationStatus::kValid, buffer_index, packet_index};
      }
    }
    return {PacketLocationStatus::kUnresolved, -1, 0};
  }

  // GPUDirect ingest support (see ProcessorStateBase::resolve_gpu_slot's doc
  // comment). Reuses fpga_index_lut/fpga_ids and locate_packet() exactly as
  // the reactive path does -- this is the second consumer of
  // initialize_buffers()'s output the plan describes, proactive instead of
  // reactive.
  bool resolve_gpu_slot(uint64_t sample_count, uint32_t fpga_id, uint16_t freq_channel,
                        void *&out_samples_addr, void *&out_scales_addr) override {
    int fpga_index_i =
        fpga_id < fpga_index_lut.size() ? fpga_index_lut[fpga_id] : -1;
    if (fpga_index_i < 0) [[unlikely]] {
      const auto it = fpga_ids.find(fpga_id);
      if (it == fpga_ids.end()) return false;
      fpga_index_i = it->second;
    }
    const size_t fpga_index = static_cast<size_t>(fpga_index_i);

    const int channel = static_cast<int>(freq_channel) - static_cast<int>(MIN_FREQ_CHANNEL);
    if (channel < 0 || channel >= static_cast<int>(T::NR_FPGA_CHANNELS)) return false;

    const auto loc = locate_packet(sample_count, fpga_index);
    if (loc.status != PacketLocationStatus::kValid) return false;
    if (!pipeline_) return false;

    auto *samples_base = static_cast<typename T::InputPacketSamplesType *>(
        pipeline_->gpu_landing_samples_ptr(loc.buffer_index));
    auto *scales_base = static_cast<typename T::PacketScalesType *>(
        pipeline_->gpu_landing_scales_ptr(loc.buffer_index));
    if (!samples_base || !scales_base) return false; // CPU-memory pipeline, not GPUDirect

    out_samples_addr = &(*samples_base)[channel][loc.packet_index + 1][fpga_index];
    const size_t receiver_index = fpga_index * T::NR_RECEIVERS_PER_PACKET;
    out_scales_addr = &(*scales_base)[channel][loc.packet_index + 1][receiver_index];
    return true;
  }
  size_t gpu_samples_slot_bytes() const override {
    return sizeof(typename T::PacketDataStructure);
  }
  size_t gpu_scales_slot_bytes() const override {
    return sizeof(typename T::PacketScaleStructure);
  }

  __attribute__((hot)) void copy_data_to_input_buffer_if_able(
      ProcessedPacket<typename T::PacketScaleStructure,
                      typename T::PacketDataStructure> &pkt,
      const int current_read_index,
      const std::array<uint64_t, T::NR_FPGA_SOURCES> &global_max) {
    int fpga_index_i =
        pkt.fpga_id < fpga_index_lut.size() ? fpga_index_lut[pkt.fpga_id] : -1;
    if (fpga_index_i < 0) [[unlikely]] {
      const auto fpga_it = fpga_ids.find(pkt.fpga_id);
      if (fpga_it == fpga_ids.end()) {
        ERROR_LOG("FPGA ID {} not found.", pkt.fpga_id);
        throw std::out_of_range("FPGA ID not found. Check logs.");
      }
      fpga_index_i = fpga_it->second;
    }
    const size_t fpga_index = static_cast<size_t>(fpga_index_i);

    const int freq_channel = pkt.freq_channel - MIN_FREQ_CHANNEL;

    const int current_buf = current_buffer;
    const uint64_t sample_count = pkt.sample_count;
    // on the first run global_max will not be set initially so will be 0.
    // We don't want it to seize up on this.
    if (sample_count > global_max[fpga_index] && global_max[fpga_index] > 0) {
      packets_future_queued.fetch_add(1, std::memory_order_relaxed);
      std::lock_guard lock(future_packet_queue_mutex);
      future_packet_queue[fpga_index].push(
          {current_read_index, sample_count, freq_channel});
      return;
    }

    // Completion watermark.  This bump must happen AFTER the future-queue
    // deferral above, not before: a deferred packet's data is not in any
    // buffer yet, and if it advanced the watermark here then the moment its
    // window rotates into existence check_buffer_completion() could complete
    // that buffer off the pre-bumped watermark -- before the periodic drain
    // re-injects the packet -- and its data would be silently counted missing
    // and zero-filled.  A single in-order producer never hits that race, but
    // concurrent capture threads (nr_capture_threads > 0, live multi-queue
    // NICs) and any bounded skew between streams do.  Deferred packets bump
    // the watermark when the drain re-processes them through the copy path
    // below; genuinely jumped streams (never drainable) are force-completed by
    // the stall safety net in drain_future_packets().
    atomic_max_u64(latest_packet_received[freq_channel][fpga_index],
                   sample_count);

    bool is_extended = false;
    int num_copied = 0;
    // copy to correct place or leave it.
    for (int buffer_num = 0; buffer_num < NR_INPUT_BUFFERS; ++buffer_num) {
      const int buffer_index = (current_buf + buffer_num) % NR_INPUT_BUFFERS;
      int packet_index =
          packet_index_for_buffer(sample_count, fpga_index, buffer_index);

      // packet_index -1 is the useful inter-FPGA overlap packet. Anything
      // earlier predates the one-shot timeline chosen by the first valid
      // packet and may be discarded; buffer boundaries are never rewritten.
      if (buffer_num == 0 && packet_index < -1) [[unlikely]] {
        // This means that this packet is less than the lowest possible
        // start token. Maybe an out-of-order packet that's coming in?
        // Regardless we can't do anything with this.
        INFO_LOG("Discarding packet as it is before current buffer with "
                 "begin_seq {} actually has packet_index {}",
                 buffers[buffer_index].start_seq[fpga_index], pkt.sample_count);
        packets_discarded.fetch_add(1);
        pkt.original_packet_processed->store(true, std::memory_order_release);
        return;
      }

      // we extend here to allow for buffer packets
      if (packet_index >= -1 &&
          packet_index < static_cast<int>(NR_PACKETS_FOR_CORRELATION) + 1) {
        const int receiver_index = fpga_index * T::NR_RECEIVERS_PER_PACKET;
        // DEBUG_LOG("Copying data to packet_index {} and channel index {} and "
        //           "receiver_index {} of buffer {}",
        //           packet_index, freq_channel, receiver_index, buffer_index);

        auto &buffer = d_samples[buffer_index];
        // we need to add 1 to the packet index to allow for the
        // packet at the front which is technically not part of the
        // correlation block.
        auto &samples =
            (*buffer->samples)[freq_channel][packet_index + 1][fpga_index];
        auto &scales =
            (*buffer->scales)[freq_channel][packet_index + 1][receiver_index];
        auto &arrival =
            buffer->arrivals[0][freq_channel][packet_index + 1][fpga_index];
        // Samples (2.5 KB for the default shape, always 64 B-aligned dest):
        // streamed past the cache — the CPU never reads this buffer, the GPU
        // DMAs it.  Scales are 40 B and stride-unaligned; plain memcpy.
        copy_nt(&samples, pkt.payload->data,
                sizeof(typename T::PacketDataStructure));
        std::memcpy(&scales, pkt.payload->scales,
                    sizeof(typename T::PacketScaleStructure));
        arrival = true;
        num_copied += 1;

        if (((packet_index == -1) && (buffer_num > 0)) ||
            ((packet_index == 0) && (buffer_num > 0)) ||
            packet_index == NR_PACKETS_FOR_CORRELATION - 1 ||
            packet_index == NR_PACKETS_FOR_CORRELATION) {
          is_extended = true;
        }
        // DEBUG_LOG("Setting original_packet_processed as true...");
        // DEBUG_LOG("original_packet_processed_before={}",
        //           *pkt.original_packet_processed);

        if (num_copied >= 1 + is_extended) {
          pkt.original_packet_processed->store(true, std::memory_order_release);
          // DEBUG_LOG("DEBUG: original_packet_processed_after={}",
          //           *pkt.original_packet_processed);

          return;
        }
      }
    }
    // sample_count <= global_max[fpga_index], so this packet wasn't deferred
    // to future_packet_queue, but it also didn't land within any of the
    // NR_INPUT_BUFFERS windows above (e.g. it's an out-of-order packet
    // belonging to a buffer that has already been released/rotated past).
    // Mark it processed so its ring slot stays reusable -- otherwise
    // *pkt.original_packet_processed stays false forever, and once enough of
    // these accumulate get_next_write_index() can no longer find a free slot.
    packets_stuck_unprocessed.fetch_add(1, std::memory_order_relaxed);
    *pkt.original_packet_processed = true;
  };
  void process_all_available_packets() {
    // Testing path — single-threaded drain of the ring.
    std::array<uint64_t, T::NR_FPGA_SOURCES> global_max{};
    for (int i = 0; i < T::NR_FPGA_SOURCES; ++i) {
      global_max[i] = global_max_end_seq[i].load(std::memory_order_acquire);
    }
    while (read_index.load() != write_index.load()) {
      int current_read_index = read_index.load();
      auto *slot = d_packet_data[current_read_index];
      // Slot reserved but not yet committed (producer still memcpy-ing) — wait.
      if (!slot->committed.load(std::memory_order_acquire)) [[unlikely]] {
        _mm_pause();
        continue;
      }
      process_packet_data(slot, current_read_index, global_max);
      read_index.store((current_read_index + 1) % RING_BUFFER_SIZE,
                       std::memory_order_release);
    }
  }

  void initialize_buffers(const uint64_t first_count, const uint32_t fpga_id) {
    INFO_LOG("[BufferInitialization] First count for FPGA ID {} was {}...",
             fpga_id, first_count);
    std::lock_guard lock(buffer_index_mutex);

    const int fpga_index = fpga_ids[fpga_id];
    const int64_t fpga_delay = fpga_delays_packet_aligned[fpga_index];
    const int64_t canonical_first =
        static_cast<int64_t>(first_count) - fpga_delay;

    // Initialization is a one-time operation. Clearing defensively prevents
    // stale/duplicate ordering entries if a test or future recovery path
    // explicitly invokes it again before any buffer has been released.
    while (!buffer_ordering_queue.empty()) buffer_ordering_queue.pop();

    for (auto i = 0; i < NR_INPUT_BUFFERS; ++i) {
      for (auto j = 0; j < T::NR_FPGA_SOURCES; ++j) {
        // need to minus the delay for whichever FPGA the reference is, then add
        // the delay for the alveo this is.
        const int64_t start_seq =
            canonical_first + fpga_delays_packet_aligned[j] +
            static_cast<int64_t>(i * NR_PACKETS_FOR_CORRELATION *
                                 NR_BETWEEN_SAMPLES);
        const int64_t end_seq =
            canonical_first + fpga_delays_packet_aligned[j] +
            static_cast<int64_t>(((i + 1) * NR_PACKETS_FOR_CORRELATION - 1) *
                                 NR_BETWEEN_SAMPLES);
        if (start_seq < 0 || end_seq < 0) {
          throw std::runtime_error(
              "Configured FPGA delays produce a negative packet sequence");
        }
        buffers[i].start_seq[j] = static_cast<uint64_t>(start_seq);
        buffers[i].end_seq[j] = static_cast<uint64_t>(end_seq);
        buffers[i].is_ready = true;
        INFO_LOG(
            "[BufferInitialization] Buffer {} for FPGA {} goes from {} to {}",
            i, j, buffers[i].start_seq[j], buffers[i].end_seq[j]);
        DEBUG_LOG("[BufferInitialization] Buffer initialization had common "
                  "delay {} and other delay {} ",
                  fpga_delay, fpga_delays_packet_aligned[j]);
        // we know 0 will be the first one - so no need to add zero
        if (i != 0 && j == 0) {
          // can just use the first FPGA for checking which buffer should
          // come next.
          buffer_ordering_queue.push({i, buffers[i].start_seq[0]});
        };
      }
    }
    for (auto j = 0; j < T::NR_FPGA_SOURCES; ++j) {
      global_max_end_seq[j].store(buffers[NR_INPUT_BUFFERS - 1].end_seq[j],
                                  std::memory_order_release);
    }
  };

  // `processed_accum`, when non-null, receives the processed-packet count in a
  // thread-local so concurrent workers don't each do a per-packet fetch_add on
  // the single global `packets_processed` atomic (that cacheline ping-ponging
  // across workers was capping useful worker scaling).  Workers pass a local
  // and flush it to the atomic once per dispatched slice; the single-threaded
  // callers (future-queue drain, process_all_available_packets) pass null and
  // increment the atomic directly, keeping the exact counts the tests assert.
  __attribute__((hot)) void process_packet_data(
      typename T::PacketEntryType *pkt, const int current_read_index,
      const std::array<uint64_t, T::NR_FPGA_SOURCES> &global_max,
      uint64_t *processed_accum = nullptr) {

    // This is where you'd do your actual processing
    // For now, just print the info and simulate some work

    if (pkt->processed.load(std::memory_order_relaxed)) [[unlikely]] {
      return;
    }

    // Non-virtual call — allows the compiler to inline parse() which carries
    // __attribute__((flatten)) and eliminates the vtable dispatch overhead.
    using ConcreteEntry = typename T::PacketEntryType;
    ProcessedPacket parsed =
        static_cast<ConcreteEntry *>(pkt)->ConcreteEntry::parse();
    // DEBUG_LOG(
    //     "Processing packet: sample_count={}, freq_channel={}, fpga_id={}, "
    //     "payload={} bytes",
    //     parsed.sample_count, parsed.freq_channel, parsed.fpga_id,
    //     parsed.payload_size);

    // DEBUG_LOG("First data point...{} + {} i",
    //           parsed.payload->data[0][0][0].real(),
    //           parsed.payload->data[0][0][0].imag());
    if (parsed.sample_count == 0) [[unlikely]] {
      throw std::runtime_error(
          "parsed sample count was zero - something has gone wrong!");
    }

    // freq_channel comes straight off the wire (CustomHeader) and is
    // otherwise unvalidated: a packet for a channel outside the configured
    // [MIN_FREQ_CHANNEL, MIN_FREQ_CHANNEL + NR_CHANNELS) window would
    // underflow/overflow this index and write out of bounds into
    // latest_packet_received / modified_since_last_completion_check / the
    // per-buffer samples-scales-arrivals arrays in
    // copy_data_to_input_buffer_if_able. Drop such packets here.
    //
    // NOTE: this check fires BEFORE buffer initialization (below). If
    // every incoming packet is out-of-range, initialize_buffers() is
    // never called, buffers never complete, and the Discarded counter
    // climbs while Processed stays zero. The most common cause is a
    // wrong --min_freq_channel value: the packets carry the actual radio
    // channel number, which must lie in [min, min+NR_CHANNELS).
    const int freq_channel = static_cast<int>(parsed.freq_channel) -
                             static_cast<int>(MIN_FREQ_CHANNEL);
    if (freq_channel < 0 || freq_channel >= static_cast<int>(T::NR_FPGA_CHANNELS))
        [[unlikely]] {
      if (!channel_discard_warned.exchange(true, std::memory_order_relaxed)) {
        WARN_LOG("Discarding packets: freq_channel={} is outside the configured "
                 "window [{}, {}). Buffer initialization will never happen while "
                 "all packets are out of range. Check --min_freq_channel.",
                 parsed.freq_channel, MIN_FREQ_CHANNEL,
                 MIN_FREQ_CHANNEL + T::NR_FPGA_CHANNELS);
        std::cout << "[ProcessorState] WARNING: freq_channel="
                  << parsed.freq_channel << " outside window ["
                  << MIN_FREQ_CHANNEL << ", "
                  << MIN_FREQ_CHANNEL + T::NR_FPGA_CHANNELS
                  << "). All packets will be discarded. "
                     "Check --min_freq_channel.\n";
      }
      packets_discarded.fetch_add(1, std::memory_order_relaxed);
      parsed.original_packet_processed->store(true, std::memory_order_release);
      return;
    }

    if (!buffer_init_flag.load(std::memory_order_acquire)) [[unlikely]] {
      // The first valid packet defines a common observation time. The exact
      // configured FPGA delays translate that one reference into the four
      // corresponding raw counter values. Initialize once and never rewrite
      // live buffer boundaries: a few packets that predate this startup point
      // may be discarded, but all streams converge naturally within their
      // small initial skew and steady-state processing remains coherent.
      std::lock_guard<std::mutex> lock(buffer_bootstrap_mutex);
      if (!buffer_init_flag.load(std::memory_order_relaxed)) {
        INFO_LOG("Initializing buffers once from first valid packet: FPGA {} "
                 "sample_count={}",
                 parsed.fpga_id, parsed.sample_count);
        initialize_buffers(parsed.sample_count, parsed.fpga_id);
        buffer_init_flag.store(true, std::memory_order_release);
      }
    }
    copy_data_to_input_buffer_if_able(parsed, current_read_index, global_max);
    if (*parsed.original_packet_processed) [[likely]] {
      if (processed_accum) {
        ++*processed_accum;
      } else {
        packets_processed.fetch_add(1, std::memory_order_relaxed);
      }
    }
    modified_since_last_completion_check[freq_channel].store(
        true, std::memory_order_release);
  };

  void
  get_global_max_packet_array(std::array<uint64_t, T::NR_FPGA_SOURCES> &arr) {
    for (int i = 0; i < T::NR_FPGA_SOURCES; ++i) {
      arr[i] = global_max_end_seq[i].load(std::memory_order_acquire);
    }
  }

  void execute_processing_pipeline_on_buffer(const int buffer_index) {};

  __attribute__((hot)) void release_buffer(const int buffer_index) {
    // Zero arrivals before taking the lock. Safe: the GPU just finished with
    // this slot, and no processor thread can claim it as current_buffer until
    // is_ready=true and the queue push happen below under buffer_index_mutex.
    std::memset(d_samples[buffer_index]->arrivals, 0,
                T::NR_FPGA_CHANNELS * (T::NR_PACKETS_FOR_CORRELATION + 2) *
                    T::NR_FPGA_SOURCES * sizeof(bool));

    {
      std::lock_guard<std::mutex> lock(buffer_index_mutex);

      std::array<uint64_t, T::NR_FPGA_SOURCES> max_end_seq_in_buffers{};
      get_global_max_packet_array(max_end_seq_in_buffers);
      const int buf_idx = buffer_index;
      auto &buffer = buffers[buf_idx];
      for (int i = 0; i < T::NR_FPGA_SOURCES; ++i) {
        const uint64_t new_start =
            max_end_seq_in_buffers[i] + 1 * NR_BETWEEN_SAMPLES;
        const uint64_t new_end =
            new_start + (NR_PACKETS_FOR_CORRELATION - 1) * NR_BETWEEN_SAMPLES;
        buffer.start_seq[i] = new_start;
        buffer.end_seq[i] = new_end;
        global_max_end_seq[i].store(new_end, std::memory_order_release);
      }

      buffer.is_populated.reset();
      buffer.is_ready = true;
      buffer_ordering_queue.push({buf_idx, buffer.start_seq[0]});
    }
    buffer_available_cv.notify_one();
    // INFO_LOG("[ProcessorState] added buffer index {} back to queue with "
    //          "start_seq {}",
    //          buffer_index, buffers[buffer_index].start_seq);
  };
  __attribute__((hot)) int advance_to_next_buffer() {

    // Move to next buffer
    // This is not necessarily the next buffer as the buffers can be
    // updated in any order. This will be the buffer that has the next
    // highest start_seq.
    // INFO_LOG("advancing to next buffer...");
    BufferOrder b;
    std::unique_lock<std::mutex> lock(buffer_index_mutex);
    buffer_available_cv.wait(lock,
                             [this] { return !buffer_ordering_queue.empty(); });
    b = buffer_ordering_queue.top();
    buffer_ordering_queue.pop();
    lock.unlock();
    // INFO_LOG("Current buffer start seq is {}", current_buffer_start_seq);
    current_buffer = b.index;

    // INFO_LOG(
    //     "next current_buffer is {} and has start_seq {} and is it ready? {}",
    //     current_buffer, buffers[current_buffer].start_seq,
    //     buffers[current_buffer].is_ready);

    // this is kinda assuming there's an async callback that will
    // ready the buffer after the data has been transferred out.
    // This async callback will come from the GPUPipeline class.
    // Reset new current buffer
    // This is just NR_CHANNELS_DEF booleans that tell us if the current
    // buffer has all the data it needs.
    buffers[b.index].is_populated.reset();
    // Set modified to true so that all channels get checked on next
    // completion check in case there are packets that went ahead.
    for (auto &f : modified_since_last_completion_check)
      f.store(true, std::memory_order_relaxed);
    //  INFO_LOG(
    //      "Current buffer is all complete. Moving to next buffer which is
    //      #{}", current_buffer);
    //  INFO_LOG("New buffer starts at packet {} and ends at {}",
    //           buffers[current_buffer].start_seq,
    //           buffers[current_buffer].end_seq);
    return b.index;
  }

  void check_buffer_completion(std::vector<int> &buffers_complete) {

    const int current_buf = current_buffer;
    for (int i = 0; i < NR_INPUT_BUFFERS; ++i) {
      // i want current buffer to be the first one in the list to preserve
      // ordering.
      const int buf_idx = (current_buf + i) % NR_INPUT_BUFFERS;
      auto &buffer = buffers[buf_idx];
      if (!buffer.is_ready) {
        continue;
      }
      const std::array<uint64_t, T::NR_FPGA_SOURCES> end_seq = buffer.end_seq;
      for (auto channel = 0; channel < T::NR_FPGA_CHANNELS; ++channel) {
        if (buffer.is_populated[channel] ||
            !modified_since_last_completion_check[channel].load(
                std::memory_order_acquire)) {
          continue;
        }
        bool all_fpgas_complete = true;
        for (int fpga = 0; fpga < T::NR_FPGA_SOURCES; ++fpga) {
          // we wait for halfway through the next buffer to be complete to avoid
          // missing out of order packets.
          const uint64_t latest = latest_packet_received[channel][fpga].load(
              std::memory_order_acquire);
          const uint64_t threshold = end_seq[fpga] + NR_BETWEEN_SAMPLES / 2;
          if (latest < threshold) {
            all_fpgas_complete = false;
            if (i == 0) {
              // TEMP DIAGNOSTIC: rate-limited dump of exactly which
              // (channel, fpga) is blocking buffer-0 completion, and by how
              // much, when this happens repeatedly without ever resolving.
              // See docs/ibverbs-zerocopy-handoff.md 32ch investigation.
              static std::chrono::steady_clock::time_point last_stall_log{};
              const auto now = std::chrono::steady_clock::now();
              if (now - last_stall_log > std::chrono::seconds(1)) {
                last_stall_log = now;
                INFO_LOG("[StallDiag] buf_idx={} channel={} fpga={} "
                         "latest_packet_received={} threshold={} gap={} "
                         "end_seq={}",
                         buf_idx, channel, fpga, latest, threshold,
                         static_cast<int64_t>(threshold) -
                             static_cast<int64_t>(latest),
                         end_seq[fpga]);
              }
              return;
            }
            break;
          }
        }
        if (all_fpgas_complete) {
          buffer.is_populated[channel] = true;
        }
      }

      if (buffer.is_populated.all()) {
        buffers_complete.push_back(buf_idx);
      }
    }
    for (int channel = 0; channel < T::NR_FPGA_CHANNELS; channel++) {
      modified_since_last_completion_check[channel].store(
          false, std::memory_order_relaxed);
    }
  };

  // How long a stream's future-queued packets may sit stuck beyond the buffer
  // horizon before the stall safety net force-completes (see
  // handle_future_stall).  The delay is the hysteresis that distinguishes a
  // genuine stream jump (sustained packet loss -- buffers stop completing, the
  // queue stays undrainable) from benign bounded skew between capture threads,
  // which resolves within microseconds as buffers release and the horizon
  // advances.  Default ~6 production buffer periods (16.5 ms each).  Runtime
  // member (not constexpr) so tests can shrink it to exercise the recovery
  // path without real sleeps.
  std::chrono::milliseconds future_stall_force_threshold{100};

  // Stall safety net.  Requires future_packet_queue_mutex to be held.
  // First sighting of a stuck-beyond-horizon queue arms a timer; if the queue
  // is still stuck past future_stall_force_threshold, bump the completion
  // watermark for every queued packet's (channel, fpga) so the stalled
  // buffers complete (holes zero-filled), release, and the horizon chases the
  // jumped stream until the queue becomes drainable -- the same loss-recovery
  // outcome the old parse-time watermark bump provided, but gated so it can
  // never race a benignly-skewed packet out of its buffer.
  void handle_future_stall(const int fpga_index) {
    const auto now = std::chrono::steady_clock::now();
    if (!future_stuck_since_valid[fpga_index]) {
      future_stuck_since[fpga_index] = now;
      future_stuck_since_valid[fpga_index] = true;
      return;
    }
    if (now - future_stuck_since[fpga_index] < future_stall_force_threshold) {
      return;
    }
    INFO_LOG("[FutureStall] FPGA {} future queue stuck beyond horizon with {} "
             "packets -- force-advancing watermarks to recover",
             fpga_index, future_packet_queue[fpga_index].size());
    std::vector<PacketOrder> stashed;
    stashed.reserve(future_packet_queue[fpga_index].size());
    while (!future_packet_queue[fpga_index].empty()) {
      stashed.push_back(future_packet_queue[fpga_index].top());
      future_packet_queue[fpga_index].pop();
    }
    for (const auto &p : stashed) {
      atomic_max_u64(latest_packet_received[p.freq_channel][fpga_index],
                     p.packet_num);
      modified_since_last_completion_check[p.freq_channel].store(
          true, std::memory_order_release);
      future_packet_queue[fpga_index].push(p);
    }
    future_stuck_since_valid[fpga_index] = false;
  }

  // Drain future-queued packets that have come into the horizon
  // (packet_num <= global_max) back through process_packet_data(), and run
  // the stall safety net for streams stuck beyond it.  Called from the
  // processor loop every iteration; try_to_lock so it never blocks capture
  // threads that are mid-deferral.
  void drain_future_packets(
      const std::array<uint64_t, T::NR_FPGA_SOURCES> &global_max) {
    std::unique_lock<std::mutex> lock(future_packet_queue_mutex,
                                      std::try_to_lock);
    if (!lock.owns_lock()) {
      return;
    }
    for (int i = 0; i < T::NR_FPGA_SOURCES; ++i) {
      while (!future_packet_queue[i].empty()) {
        auto pkt = future_packet_queue[i].top();
        if (pkt.packet_num > global_max[i]) {
          handle_future_stall(i);
          break;
        }
        future_stuck_since_valid[i] = false;
        future_packet_queue[i].pop();
        lock.unlock();
        auto *entry = d_packet_data[pkt.index];
        if (entry->length > 0 &&
            entry->committed.load(std::memory_order_acquire) &&
            !entry->processed.load(std::memory_order_relaxed)) {
          process_packet_data(entry, pkt.index, global_max);
        }
        lock.lock();
      }
      if (future_packet_queue[i].empty()) {
        future_stuck_since_valid[i] = false;
      }
    }
  }

  __attribute__((hot)) void process_packets() {

    // using clock = std::chrono::high_resolution_clock;
    // auto cpu_start = clock::now();
    // auto cpu_end = clock::now();
    INFO_LOG("Processor thread started");
    INFO_LOG("Listening for freq_channels [{}, {}), NR_FPGA_SOURCES={}",
             MIN_FREQ_CHANNEL, MIN_FREQ_CHANNEL + T::NR_FPGA_CHANNELS,
             T::NR_FPGA_SOURCES);
    std::cout << "[ProcessorState] Listening for freq_channels ["
              << MIN_FREQ_CHANNEL << ", "
              << MIN_FREQ_CHANNEL + T::NR_FPGA_CHANNELS
              << "), NR_FPGA_SOURCES=" << T::NR_FPGA_SOURCES << "\n";
    start_processing_threads();
    int current_read_index;
    constexpr int num_loops_before_completion_check = 1;
    int packets_until_completion_check = num_loops_before_completion_check;
    current_read_index = read_index.load(std::memory_order_relaxed);

    // Strided slot ownership (thread t owns ring slots t, t+N, t+2N, ...) is
    // only DISJOINT between capture threads when N divides the ring size:
    // slot = linear % RING_BUFFER_SIZE, and for stride ∤ ring the residues
    // interleave, so two threads whose claim rates drift more than a ring
    // apart end up claiming the SAME slots and serialise on each other's
    // processed flags (observed as multi-second stalls under uneven per-queue
    // packet rates).  Warn loudly rather than assert: existing single-thread
    // captures are unaffected.
    if (nr_capture_threads > 0 &&
        RING_BUFFER_SIZE % static_cast<size_t>(nr_capture_threads) != 0) {
      ERROR_LOG("RING_BUFFER_SIZE ({}) is not divisible by nr_capture_threads "
                "({}): strided capture threads will share ring slots and can "
                "stall each other under uneven packet rates. Use a divisible "
                "ring size.",
                RING_BUFFER_SIZE, nr_capture_threads);
    }

    // Per-thread local read positions for the strided consumer path.
    // Initialized to tid because thread tid's linear sequence starts at tid,
    // not 0 (matches KernelSocketPacketCapture's my_linear = thread_id_).
    uint64_t per_thread_my_read[ProcessorStateBase::MAX_CAPTURE_THREADS];
    for (int tid = 0; tid < ProcessorStateBase::MAX_CAPTURE_THREADS; ++tid)
      per_thread_my_read[tid] = static_cast<uint64_t>(tid);

    std::array<uint64_t, T::NR_FPGA_SOURCES> global_max{};
    // Upper bound on how many slots a single dispatch_and_wait() call below
    // may be asked to process for one capture thread's turn in the
    // round-robin `for (tid...)` loop. dispatch_and_wait() blocks until
    // every slot in its batch is committed, and per_thread_read_linear[tid]
    // (this thread's published *progress*, as opposed to per_thread_claim,
    // its published *reservation* watermark) only advances once the whole
    // call returns -- so a large cap here does not just risk one slow
    // thread starving the other three of a turn (the case the surrounding
    // per-thread loop's own comment already calls out), it means progress
    // is only ever visible in large, infrequent jumps.
    // For KernelSocketPacketCapture this was rarely the limiting factor:
    // reservation there happens synchronously with the recvmmsg batch that
    // fills it (capped at BATCH_SIZE=256 packets), so per_thread_claim
    // naturally tracks real arrival and rarely approaches this cap.
    // LibibverbsPacketCapture's zero-copy path reserves active_frames_
    // (~1024) receive buffers up front to keep the RDMA pipeline fed, and
    // then continuously reserves one replacement per completion regardless
    // of how far ahead of actual processing that puts it -- confirmed by
    // direct measurement (2026-09-20) that with the 6000 cap, a single
    // dispatch_and_wait() call for one thread's ~1500-slot backlog was
    // still not complete after 30+ seconds, during which the other three
    // threads' already-committed data went completely unprocessed and
    // per_thread_read_linear (hence Stats' "Processed" counter) never
    // advanced from zero. Reducing the cap to match KernelSocketPacketCapture's
    // own natural batch granularity fixes this for zero-copy while being a
    // no-op for the kernel-socket path (whose batches essentially never
    // reach this size in the first place).
    constexpr int REGULAR_BATCH_SIZE = 256;
    uint64_t dispatch_generation = 0;

    // Distribute `slots` items (each `stride` ring-index steps apart, starting
    // at `base_ring`) across the processor thread plus WORKER_COUNT helpers.
    // The processor used to spin while every byte was copied by the helpers;
    // making it process one slice adds useful copy bandwidth without adding a
    // thread or another task-completion synchronization.
    // Write all task ranges before any release store so workers see consistent
    // state from their acquire load on worker_has_task.
    auto dispatch_and_wait = [&](int base_ring, int slots, int stride) {
      using DispatchClock = std::chrono::steady_clock;
      const auto dispatch_start = profile_worker_dispatch ? DispatchClock::now()
                                                          : DispatchClock::time_point{};
      const uint64_t generation = ++dispatch_generation;
      const int lane_count = WORKER_COUNT + 1;
      const int per_lane = (slots + lane_count - 1) / lane_count;
      const int main_items = std::min(per_lane, slots);
      const WorkRange main_task = {
          base_ring,
          (base_ring + main_items * stride) % (int)RING_BUFFER_SIZE,
          stride};
      int items_done = main_items;
      int workers_with_tasks = 0;
      for (int i = 0; i < WORKER_COUNT; ++i) {
        const int items = std::min(per_lane, slots - items_done);
        if (items == 0) break;
        const int w_start =
            (base_ring + items_done * stride) % (int)RING_BUFFER_SIZE;
        const int w_end =
            (base_ring + (items_done + items) * stride) % (int)RING_BUFFER_SIZE;
        worker_tasks[i] = {w_start, w_end, stride};
        ++workers_with_tasks;
        items_done += items;
      }
      if (use_worker_mailboxes) {
        for (int i = 0; i < workers_with_tasks; ++i)
          worker_mailboxes[i].requested.store(generation, std::memory_order_release);
      } else {
        num_workers_with_tasks.store(workers_with_tasks,
                                     std::memory_order_relaxed);
        for (int i = 0; i < workers_with_tasks; ++i)
          worker_has_task[i].store(true, std::memory_order_release);
      }
      const uint64_t main_processed = process_work_range(main_task);
      if (main_processed)
        packets_processed.fetch_add(main_processed, std::memory_order_relaxed);
      const auto wait_start = profile_worker_dispatch ? DispatchClock::now()
                                                      : DispatchClock::time_point{};
      // Wait for the workers to actually FINISH the dispatched slices.
      // Breaking out early on !running (the old behaviour) let the
      // completion check below run concurrently with in-flight worker
      // copies at shutdown, so a buffer could complete off one worker's
      // watermark while another worker's arrival flags weren't written yet
      // -- real arrivals then counted missing and zero-filled.  Workers are
      // guaranteed to terminate their slices even at shutdown (their
      // committed-spin bails on !running), so only stop waiting once every
      // worker has exited (workers_alive == 0) and its pending decrements
      // can no longer come.
      if (use_worker_mailboxes) {
        for (int i = 0; i < workers_with_tasks; ++i) {
          while (worker_mailboxes[i].completed.load(std::memory_order_acquire) != generation) {
            // A worker may exit just before a shutdown-time dispatch. An
            // exited worker cannot have an in-flight copy; active workers
            // must still finish before completion checks run.
            if (worker_mailboxes[i].exited.load(std::memory_order_acquire))
              break;
            _mm_pause();
          }
        }
      } else {
        // TEMP DIAGNOSTIC: rate-limited heartbeat so a stall inside this
        // otherwise-silent spin is visible instead of indistinguishable
        // from process_packets() simply not being called at all. See
        // docs/ibverbs-zerocopy-handoff.md 32ch investigation.
        auto wait_diag_start = std::chrono::steady_clock::now();
        auto last_wait_diag = wait_diag_start;
        while (num_workers_with_tasks.load(std::memory_order_acquire) > 0) {
          if (!running.load(std::memory_order_relaxed) &&
              workers_alive.load(std::memory_order_acquire) == 0)
            break;
          const auto now = std::chrono::steady_clock::now();
          if (now - last_wait_diag > std::chrono::seconds(1)) {
            last_wait_diag = now;
            std::cerr << "DIAG dispatch_and_wait: stuck waiting for workers "
                         "for "
                      << std::chrono::duration_cast<std::chrono::seconds>(
                             now - wait_diag_start)
                             .count()
                      << "s, num_workers_with_tasks="
                      << num_workers_with_tasks.load() << " gen="
                      << generation << std::endl;
          }
          _mm_pause();
        }
      }
      if (profile_worker_dispatch) {
        const auto end = DispatchClock::now();
        ++dispatch_batches;
        dispatch_slots += slots;
        dispatch_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - wait_start).count();
        dispatch_total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - dispatch_start).count();
      }
    };

    while (running.load(std::memory_order_acquire)) [[likely]] {
      if (nr_capture_threads > 0) {
        // Strided mode: each capture thread owns slots tid, tid+N, tid+2N, ...
        // Process each thread's claimed range independently so a slow or idle
        // thread never stalls the others.
        const int stride = nr_capture_threads;
        bool any_processed = false;

        for (int tid = 0; tid < nr_capture_threads; ++tid) {
          const uint64_t claim =
              per_thread_claim[tid].linear.load(std::memory_order_acquire);
          if (claim <= per_thread_my_read[tid]) continue;

          const int slots = static_cast<int>(
              std::min((claim - per_thread_my_read[tid]) / (uint64_t)stride,
                       (uint64_t)REGULAR_BATCH_SIZE));
          if (slots == 0) continue;

          dispatch_and_wait(
              static_cast<int>(per_thread_my_read[tid] % RING_BUFFER_SIZE),
              slots, stride);

          per_thread_my_read[tid] += (uint64_t)slots * stride;
          per_thread_read_linear[tid].linear.store(per_thread_my_read[tid],
                                                   std::memory_order_release);
          any_processed = true;
        }

        if (!any_processed) [[unlikely]] {
          _mm_pause();
          continue;
        }
      } else {
        // Legacy mode: single producer advances write_index under mutex.
        const int current_write_index =
            write_index.load(std::memory_order_acquire);
        const int available =
            (current_write_index - current_read_index + (int)RING_BUFFER_SIZE) %
            (int)RING_BUFFER_SIZE;

        if (available <= 0) [[unlikely]] {
          _mm_pause();
          continue;
        }

        const int to_process = std::min(available, REGULAR_BATCH_SIZE);
        const int slice_end =
            (current_read_index + to_process) % (int)RING_BUFFER_SIZE;

        dispatch_and_wait(current_read_index, to_process, 1);

        current_read_index = slice_end;
        read_index.store(current_read_index, std::memory_order_release);
      }

      if (--packets_until_completion_check == 0) {
        // Before checking buffer completion drain any packets from the
        // queue that should be in this buffer
        get_global_max_packet_array(global_max);
        drain_future_packets(global_max);
        handle_buffer_completion();
        packets_until_completion_check = num_loops_before_completion_check;
      }
    }
    std::cout << "Main processor thread is shutting down.";
    std::cout << " Waiting for sub-threads." << std::endl;
    stop_processing_threads();
    std::cout << "Processor thread exiting\n";
  };

  __attribute__((hot)) uint64_t process_work_range(const WorkRange &task) {
    std::array<uint64_t, T::NR_FPGA_SOURCES> global_max{};
    // A zero horizon disables future-packet deferral. With asynchronous H2D
    // recycling, producers can outrun the available windows: those packets
    // would otherwise advance completion watermarks without landing anywhere.
    // A stale, smaller snapshot is safe: the future queue retries after release.
    get_global_max_packet_array(global_max);
    auto [start, end, stride] = task;
    int idx = start;
    uint64_t local_processed = 0;

    while (idx != end) {
      auto *entry = d_packet_data[idx];

      constexpr int PREFETCH_DIST = 12;
      const int pre_idx = (idx + PREFETCH_DIST * stride) % RING_BUFFER_SIZE;
      auto *pre_entry = d_packet_data[pre_idx];
      __builtin_prefetch(pre_entry, 0, 1);
      __builtin_prefetch(pre_entry->data, 0, 1);
      __builtin_prefetch(&pre_entry->committed, 0, 1);
      __builtin_prefetch(pre_entry->data + 192, 0, 1);
      __builtin_prefetch(pre_entry->data + 768, 0, 1);
      __builtin_prefetch(pre_entry->data + 1536, 0, 1);

      bool bailed = false;
      // TEMP DIAGNOSTIC: rate-limited heartbeat -- if a specific ring slot
      // is reserved but never committed, this spins forever with no other
      // visible symptom than dispatch_and_wait() never returning. See
      // docs/ibverbs-zerocopy-handoff.md 32ch investigation.
      auto commit_wait_start = std::chrono::steady_clock::now();
      auto last_commit_diag = commit_wait_start;
      while (!entry->committed.load(std::memory_order_acquire)) [[unlikely]] {
        if (!running.load(std::memory_order_relaxed)) {
          bailed = true;
          break;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_commit_diag > std::chrono::seconds(1)) {
          last_commit_diag = now;
          std::cerr << "DIAG process_work_range: stuck waiting for slot "
                    << idx << " to commit for "
                    << std::chrono::duration_cast<std::chrono::seconds>(
                           now - commit_wait_start)
                           .count()
                    << "s (task start=" << start << " end=" << end
                    << " stride=" << stride << ")" << std::endl;
        }
        _mm_pause();
      }
      if (bailed)
        break;
      if (entry->length > 0 &&
          !entry->processed.load(std::memory_order_relaxed)) {
        process_packet_data(entry, idx, global_max, &local_processed);
      }
      idx = (idx + stride) % RING_BUFFER_SIZE;
    }

    return local_processed;
  }

  __attribute__((hot)) void worker_thread_fn(int worker_id) {
    std::cout << "Starting worker with id " << worker_id << std::endl;
    // Affinity is opt-in via SPATIAL_WORKER_CPUS="4,5,6" (one entry per
    // worker).  Unset means no pinning — the previous hard pin to cores
    // 0..N-1 landed workers on the cores NIC IRQs typically use.
    const int cpu =
        nth_cpu_from_list(std::getenv("SPATIAL_WORKER_CPUS"), worker_id);
    if (cpu >= 0) {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(cpu, &cpuset);
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
      std::cout << "Worker " << worker_id << " pinned to CPU " << cpu
                << std::endl;
    }

    if (use_worker_mailboxes) {
      auto &mailbox = worker_mailboxes[worker_id];
      uint64_t completed = 0;
      while (true) {
        const uint64_t requested = mailbox.requested.load(std::memory_order_acquire);
        if (requested == completed) {
          if (!running.load(std::memory_order_relaxed)) {
            mailbox.exited.store(true, std::memory_order_release);
            workers_alive.fetch_sub(1, std::memory_order_acq_rel);
            return;
          }
          _mm_pause();
          continue;
        }
        const uint64_t count = process_work_range(worker_tasks[worker_id]);
        if (count)
          packets_processed.fetch_add(count, std::memory_order_relaxed);
        completed = requested;
        // Publishes all copies and arrival flags. Coordinator acquires this
        // before checking buffer completion or overwriting this worker's task.
        mailbox.completed.store(completed, std::memory_order_release);
      }
    }

    while (true) {
      // Spin-wait for a task signal from the main thread, or exit.
      while (!worker_has_task[worker_id].load(std::memory_order_acquire)) {
        if (!running.load(std::memory_order_relaxed)) {
          // Publish the exit so dispatch_and_wait's shutdown path knows this
          // worker can no longer decrement num_workers_with_tasks.
          workers_alive.fetch_sub(1, std::memory_order_acq_rel);
          std::cout << "Worker " << worker_id << " exited\n";
          return;
        }
        _mm_pause();
      }

      const uint64_t local_processed =
          process_work_range(worker_tasks[worker_id]);

      if (local_processed) {
        packets_processed.fetch_add(local_processed, std::memory_order_relaxed);
      }

      // Clear flag BEFORE decrementing so the main thread can't re-signal this
      // worker and have us clear the new signal.
      worker_has_task[worker_id].store(false, std::memory_order_relaxed);
      num_workers_with_tasks.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  __attribute__((hot)) void handle_buffer_completion(bool force_flush = false) {

    if (pipeline_ == nullptr) [[unlikely]] {
      throw std::logic_error(
          "Pipeline has not been set. Ensure that set_pipeline has been "
          "called on ProcessorState class.");
    }
    using clock = std::chrono::high_resolution_clock;
    auto cpu_start = clock::now();
    auto cpu_end = clock::now();
    // INFO_LOG("Checking buffer completion...");
    std::vector<int> buffers_complete;
    check_buffer_completion(buffers_complete);

    if (force_flush && buffers_complete.empty()) {
      buffers_complete.push_back(current_buffer);
    }

    if (buffers_complete.empty()) {
      return;
    }

    int current_buf = buffers_complete[0];
    for (int i = 0; i < buffers_complete.size(); ++i) {
      INFO_LOG("pipeline feeding {}", current_buf);
      auto &buffer = buffers[current_buf];

      if (std::find(buffers_complete.begin(), buffers_complete.end(),
                    current_buf) == buffers_complete.end()) [[unlikely]] {
        return;
      }
      // INFO_LOG("Buffer is complete - passing to output pipeline...");
      //  Send off data to be processed by CUDA pipeline.
      //  Then advance to next buffer and keep iterating.

      buffer.is_ready = false;
      cpu_start = clock::now();
      // INFO_LOG("Zeroing missing packets...");
      d_samples[current_buf]->zero_missing_packets();
      packets_missing += d_samples[current_buf]->get_num_missing_packets();
      cpu_end = clock::now();
      // DEBUG_LOG("CPU time for zeroing packets: {} us",
      //           std::chrono::duration_cast<std::chrono::microseconds>(cpu_end
      //           -
      //                                                                 cpu_start)
      //               .count());
      // Use FPGA-0's window boundaries as the canonical block identifier.
      // release_buffer() will overwrite buffer.start_seq/end_seq for the next
      // cycle, so capture them now before handing off to the pipeline.
      d_samples[current_buf]->start_seq_id = buffer.start_seq[0];
      d_samples[current_buf]->end_seq_id = buffer.end_seq[0];
      //  order here is important. As the pipeline can async update
      //  the start/end seqs we need to capture the
      //  start_seq, then execute the pipeline, then advance using
      //  the saved copy of the start_seq.
      // INFO_LOG("Enqueueing buffer {} for pipeline...", current_buffer);
      {
        if (!synchronous_pipeline) [[likely]] {
          // Lock-free SPSC push: producer is always this thread (processor),
          // consumer is pipeline_feeder.  Spin on full only in the extremely
          // rare case that the GPU falls behind by PIPELINE_Q_CAP buffers.
          const uint64_t t =
              pipeline_q_tail_.v.load(std::memory_order_relaxed);
          while (t - pipeline_q_head_.v.load(std::memory_order_acquire) >=
                 PIPELINE_Q_CAP) {
            if (!running.load(std::memory_order_acquire))
              return;
            _mm_pause();
          }
          pipeline_q_buf_[t % PIPELINE_Q_CAP] = current_buf;
          pipeline_q_tail_.v.store(t + 1, std::memory_order_release);
        } else [[unlikely]] {
          pipeline_->execute_pipeline(d_samples[current_buf]);
          pipeline_runs_queued += 1;
        }
      }
      cpu_start = clock::now();
      current_buf = advance_to_next_buffer();
      cpu_end = clock::now();
    }
    // DEBUG_LOG("CPU time for advancing to next buffer: {} us",
    //           std::chrono::duration_cast<std::chrono::microseconds>(cpu_end
    //           -
    //                                                                 cpu_start)
    //               .count());
    // INFO_LOG("Done!");
  };

  void shutdown() {
    // Workers and pipeline_feeder both spin-check running and exit naturally.
    running.store(0, std::memory_order_release);
  };

  void pipeline_feeder() {
    std::cout << "Pipeline feeder starting up...\n";
    // Lock-free SPSC consumer: spin briefly on empty, then sleep to avoid
    // burning 100% CPU.  Buffer period ≈ 256/15500 s ≈ 16.5ms; a 200µs
    // sleep wastes at most ~1.2% of that period in latency.
    int spin_count = 0;
    while (running.load(std::memory_order_acquire) == 1) {
      const uint64_t h = pipeline_q_head_.v.load(std::memory_order_relaxed);
      if (h == pipeline_q_tail_.v.load(std::memory_order_acquire)) {
        if (++spin_count < 1000) {
          _mm_pause();
        } else {
          spin_count = 0;
          std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        continue;
      }
      spin_count = 0;
      const int buffer_index =
          pipeline_q_buf_[h % PIPELINE_Q_CAP];
      pipeline_q_head_.v.store(h + 1, std::memory_order_release);
      pipeline_->execute_pipeline(d_samples[buffer_index]);
      pipeline_runs_queued += 1;
    }
    std::cout << "Pipeline feeder exiting!\n";
  }

  void *get_current_write_pointer() {
    return (void *)&(d_packet_data[write_index]->data);
  }
  void *get_next_write_pointer() {
    while (!get_next_write_index() && running.load(std::memory_order_acquire)) {
      _mm_pause();
    }
    return get_current_write_pointer();
  }

  void add_received_packet_metadata(const int length,
                                    const sockaddr_in &client_addr) {
    d_packet_data[write_index]->length = length;
    d_packet_data[write_index]->sender_addr = client_addr;
    d_packet_data[write_index]->processed.store(false,
                                                std::memory_order_relaxed);
    // Release: consumer's acquire on committed synchronises-with this store,
    // ensuring it sees the data[] already memcpy'd before this call.
    d_packet_data[write_index]->committed.store(true,
                                                std::memory_order_release);
  }

  // Reserve up to max_n ring slots.  Must be called under producer_mutex
  // (brief: only index arithmetic, no data movement).  Sets committed=false on
  // each claimed slot so the consumer waits for commit_write_batch().
  int reserve_write_batch(int max_n, void **slot_ptrs,
                          int *slot_indices) override {
    int reserved = 0;
    // Slot 0 uses the current write_index (caller positioned it via a prior
    // get_next_write_pointer call, or this is the first batch).
    int cur = write_index.load(std::memory_order_relaxed);
    d_packet_data[cur]->committed.store(false, std::memory_order_relaxed);
    slot_ptrs[reserved] = get_current_write_pointer();
    slot_indices[reserved] = cur;
    reserved++;
    // Claim additional contiguous slots.
    int next = (cur + 1) % RING_BUFFER_SIZE;
    while (reserved < max_n) {
      if (next == read_index.load(std::memory_order_acquire)) {
        // Ring full: wait for the consumer to free slots, but never spin
        // past shutdown (the consumer is gone by then).
        if (!running.load(std::memory_order_acquire)) {
          break;
        }
        _mm_pause();
        continue;
      }
      if (!d_packet_data[next]->processed.load(std::memory_order_relaxed)) {
        next = (next + 1) % RING_BUFFER_SIZE; // slot still held by consumer
        continue;
      }
      d_packet_data[next]->committed.store(false, std::memory_order_relaxed);
      slot_ptrs[reserved] = (void *)&(d_packet_data[next]->data);
      slot_indices[reserved] = next;
      reserved++;
      next = (next + 1) % RING_BUFFER_SIZE;
    }

    write_index.store(next, std::memory_order_release);
    return reserved;
  }

  // Commit slots previously reserved.  No lock required: each slot is owned
  // exclusively by this producer thread from reserve until commit.  The
  // committed.store(release) pairs with the consumer's committed.load(acquire),
  // ensuring data[] and metadata are visible before the consumer processes.
  void commit_write_batch(int n, const int *slot_indices, const int *lens,
                          const sockaddr_in *addrs) override {
    for (int i = 0; i < n; ++i) {
      int s = slot_indices[i];
      d_packet_data[s]->length = lens[i];
      d_packet_data[s]->sender_addr = addrs[i];
      d_packet_data[s]->processed.store(false, std::memory_order_relaxed);
      d_packet_data[s]->committed.store(true, std::memory_order_release);
    }
  }

  // Return reserved-but-unfilled slots as empty: consumers skip them
  // (length==0, committed so they never spin) and producers can reclaim them
  // (processed==true).
  void abandon_write_batch(int n, const int *slot_indices) override {
    for (int i = 0; i < n; ++i) {
      auto *slot = d_packet_data[slot_indices[i]];
      slot->length = 0;
      slot->processed.store(true, std::memory_order_relaxed);
      slot->committed.store(true, std::memory_order_release);
    }
  }

  size_t slot_data_capacity() const override {
    return T::PacketEntryType::DATA_CAPACITY;
  }

  void *packet_ring_base() const override {
    return static_cast<void *>(d_packet_data_pool);
  }
  size_t packet_ring_bytes() const override {
    return sizeof(typename T::PacketEntryType) * RING_BUFFER_SIZE;
  }
  size_t packet_ring_slot_stride() const override {
    return sizeof(typename T::PacketEntryType);
  }
  size_t packet_ring_slot_offset() const override {
    const auto *entry = d_packet_data_pool;
    return reinterpret_cast<const uint8_t *>(&entry->data) -
           reinterpret_cast<const uint8_t *>(entry);
  }
  size_t packet_ring_slot_count() const override { return RING_BUFFER_SIZE; }

  // Lock-free strided override: no mutex, no shared write_index.
  // Thread thread_id owns slots thread_id, thread_id+N, thread_id+2N, ...
  // my_linear is the caller's monotonic counter (local to the capture thread).
  // per_thread_claim[thread_id] is published so the consumer can independently
  // track each thread's claimed slots without taking any lock.
  int reserve_write_batch_strided(int thread_id, uint64_t &my_linear,
                                  int max_n, void **slot_ptrs,
                                  int *slot_indices) override {
    const int stride = (nr_capture_threads > 0) ? nr_capture_threads : 1;
    int reserved = 0;
    for (int b = 0; b < max_n; ++b) {
      // Ring-full: spin until the consumer has advanced this thread's
      // per_thread_read_linear far enough to free this slot for reuse.
      while (my_linear -
                 per_thread_read_linear[thread_id].linear.load(
                     std::memory_order_acquire) >=
             RING_BUFFER_SIZE) {
        if (!running.load(std::memory_order_acquire))
          return reserved;
        _mm_pause();
      }
      const int slot = static_cast<int>(my_linear % RING_BUFFER_SIZE);
      // Fine-grained check: wait until the consumer (or abandon_write_batch)
      // has marked this specific slot as fully processed so we don't
      // overwrite data still held by future_packet_queue.
      while (!d_packet_data[slot]->processed.load(std::memory_order_acquire)) {
        if (!running.load(std::memory_order_acquire))
          return reserved;
        _mm_pause();
      }
      d_packet_data[slot]->committed.store(false, std::memory_order_relaxed);
      slot_ptrs[reserved] = &d_packet_data[slot]->data;
      slot_indices[reserved] = slot;
      ++reserved;
      my_linear += stride;
    }
    // Publish watermark: consumer reads per_thread_claim[tid] per-thread
    // to advance its own read cursor for this thread's slots.
    per_thread_claim[thread_id].linear.store(my_linear,
                                             std::memory_order_release);
    return reserved;
  }

private:
  void cleanup() {
    if (d_packet_data_pool != nullptr) {
      using EntryT = typename T::PacketEntryType;
      for (size_t i = 0; i < RING_BUFFER_SIZE; ++i) {
        d_packet_data_pool[i].~EntryT();
      }
      std::free(d_packet_data_pool);
    }
    d_packet_data_pool = nullptr;
    std::fill_n(d_packet_data, RING_BUFFER_SIZE, nullptr);

    for (auto i = 0; i < NR_INPUT_BUFFERS; ++i) {
      delete d_samples[i];
      d_samples[i] = nullptr;
    }
  };

  struct BufferOrder {
    int index;
    uint64_t start_seq;

    // Compare to make the priority queue a min-heap based on start_seq
    bool operator>(const BufferOrder &other) const {
      return start_seq > other.start_seq;
    }
  };

  struct PacketOrder {
    int index;
    uint64_t packet_num;
    // MIN_FREQ_CHANNEL-adjusted channel index, recorded at deferral so the
    // stall safety net in drain_future_packets() knows which watermark to
    // force-bump without re-parsing the ring slot.
    int freq_channel;

    bool operator>(const PacketOrder &other) const {
      return packet_num > other.packet_num;
    }
  };

  // Lock-free SPSC ring for completed buffer indices.
  // Producer: handle_buffer_completion (processor thread).
  // Consumer: pipeline_feeder (feeder thread).
  // Tail and head live on separate cache lines to prevent false sharing.
  static constexpr size_t PIPELINE_Q_CAP = 64;  // power of two, >> NR_INPUT_BUFFERS
  struct alignas(64) PipelineQPtr { std::atomic<uint64_t> v{0}; };
  PipelineQPtr pipeline_q_tail_;  // written by producer
  PipelineQPtr pipeline_q_head_;  // written by consumer
  int pipeline_q_buf_[PIPELINE_Q_CAP]{};
  // This is for packets that arrive but their buffer is not yet created.
  // The read pointer moves on but keep these as things to process.
  // They will not be overwritten by the write pointer as it checks whether or
  // not they have been processed.
  std::array<std::priority_queue<PacketOrder, std::vector<PacketOrder>,
                                 std::greater<PacketOrder>>,
             T::NR_FPGA_SOURCES>
      future_packet_queue;
  std::mutex future_packet_queue_mutex;
  // Stall-detection state for the drain safety net (handle_future_stall):
  // when this fpga's queue head first got stuck beyond the horizon, and
  // whether that timestamp is armed.  Only touched under
  // future_packet_queue_mutex on the processor thread.
  std::array<std::chrono::steady_clock::time_point, T::NR_FPGA_SOURCES>
      future_stuck_since{};
  std::array<bool, T::NR_FPGA_SOURCES> future_stuck_since_valid{};
  std::array<std::atomic<bool>, T::NR_FPGA_CHANNELS> modified_since_last_completion_check;
  std::priority_queue<BufferOrder, std::vector<BufferOrder>,
                      std::greater<BufferOrder>>
      buffer_ordering_queue;
  std::condition_variable buffer_available_cv;
  std::array<std::atomic<uint64_t>, T::NR_FPGA_SOURCES> global_max_end_seq{0};
  std::atomic<bool> buffer_init_flag{false};
  // Serializes the one permitted startup initialization. Buffer boundaries
  // are never re-seeded once published to packet-processing workers.
  std::mutex buffer_bootstrap_mutex;
  std::array<int64_t, T::NR_FPGA_SOURCES> fpga_delays,
      fpga_delays_packet_aligned;
  std::array<int, T::NR_FPGA_SOURCES> fpga_delays_subpacket;

  std::array<int16_t, 256> fpga_index_lut;
  // Per-worker task signaling. Main writes worker_tasks[i] then stores true
  // (release) to worker_has_task[i]. Workers spin-acquire on worker_has_task
  // and read the task without any mutex. Workers store false (relaxed) then
  // decrement num_workers_with_tasks (acq_rel) to signal completion.
  std::array<std::atomic<bool>, WORKER_COUNT> worker_has_task;
  std::array<WorkRange, WORKER_COUNT> worker_tasks;
  struct WorkerMailbox {
    alignas(64) std::atomic<uint64_t> requested{0}; // coordinator writes
    alignas(64) std::atomic<uint64_t> completed{0}; // worker writes
    std::atomic<bool> exited{false};
  };
  std::array<WorkerMailbox, WORKER_COUNT> worker_mailboxes;
  std::atomic<int> num_workers_with_tasks = 0;
  // Number of worker threads that have not yet exited; lets
  // dispatch_and_wait distinguish "workers still finishing their slices"
  // (keep waiting -- shutdown-correctness) from "workers gone, pending
  // decrements will never arrive" (stop waiting).
  std::atomic<int> workers_alive = 0;

  std::vector<std::thread> workers;

  void start_processing_threads() {
    workers_alive.store(WORKER_COUNT, std::memory_order_release);
    for (int i = 0; i < WORKER_COUNT; i++) {
      worker_has_task[i].store(false);
      worker_mailboxes[i].requested.store(0, std::memory_order_relaxed);
      worker_mailboxes[i].completed.store(0, std::memory_order_relaxed);
      worker_mailboxes[i].exited.store(false, std::memory_order_relaxed);
      workers.emplace_back(&ProcessorState::worker_thread_fn, this, i);
    };
  };
  void stop_processing_threads() {
    // Workers spin on running.load(); setting it false (done by the caller
    // before invoking this) wakes them from their idle spin.
    std::cout << "Trying to join worker threads...\n";
    for (auto &t : workers) {
      t.join();
    }
  }
};

class PacketInput {
public:
  virtual void get_packets(ProcessorStateBase &state) = 0;
  // Cumulative drop counter — populated by implementations that track drops
  // (KernelSocketPacketCapture via SO_RXQ_OVFL / VMA equivalent).  Returns 0
  // for backends that don't track drops (PCAP, ibverbs).
  virtual uint32_t get_drops() const { return 0; }

  // Optional backend initialization after the ProcessorState/ring exists.
  // Kernel and pcap inputs do nothing; ibverbs uses this hook to register the
  // packet ring and post receives directly into ring slots.
  virtual void arm_zero_copy(ProcessorStateBase & /*state*/) {}

  virtual ~PacketInput() = default;
};

class KernelSocketPacketCapture : public PacketInput {
public:
  // busy_poll_us: if > 0, enables SO_BUSY_POLL on the socket for that many
  // microseconds.  Reduces per-packet latency on dedicated cores at the cost
  // of CPU spin.  Leave at 0 on shared/virtual machines.
  KernelSocketPacketCapture(std::string &ifname, int port, int buffer_size,
                            int recv_buffer_size = 64 * 1024 * 1024,
                            int busy_poll_us = 0,
                            int thread_id = 0,
                            int nr_threads = 1);
  ~KernelSocketPacketCapture();

  void get_packets(ProcessorStateBase &state) override {
    std::cout << "Starting packet capture on ifname " << ifname
              << " (thread_id=" << thread_id_ << ")" << std::endl;

    // Optional CPU affinity for this capture thread.  Set SPATIAL_CAPTURE_CPUS
    // to a comma-separated list of CPU IDs (e.g. "0,1,2,3"), one per capture
    // thread in order.  When unset, the OS schedules freely.
    {
      const int cpu =
          nth_cpu_from_list(std::getenv("SPATIAL_CAPTURE_CPUS"), thread_id_);
      if (cpu >= 0) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(cpu, &cs);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) == 0)
          INFO_LOG("Capture thread {} pinned to CPU {}", thread_id_, cpu);
        else
          INFO_LOG("Capture thread {}: failed to pin to CPU {}", thread_id_, cpu);
      }
    }

    // Per-thread monotonic write counter for the lock-free strided path.
    // Starts at thread_id_ so each thread owns non-overlapping slots:
    // thread 0 → 0, N, 2N, ...   thread 1 → 1, N+1, 2N+1, ...  etc.
    uint64_t my_linear = static_cast<uint64_t>(thread_id_);

    // Zero-copy variant: ring slots are reserved up front and recvmmsg
    // receives *directly into slot->data*, eliminating the staging buffer
    // and one full copy of the data stream (~25 GB/s of memory traffic at
    // 5 Mpps).  Sequence per batch:
    //   1. poll() until the socket is readable — no ring slots are held
    //      while waiting, so consumers never spin on uncommitted slots.
    //   2. reserve BATCH_SIZE slots (lock-free strided, no producer_mutex).
    //   3. recvmmsg(MSG_DONTWAIT) straight into the reserved slots.
    //   4. commit the filled slots; abandon the rest as empty so neither
    //      side ever waits on them.
    static constexpr int BATCH_SIZE = 256;
    // Plain stack locals, not `static thread_local`: get_packets is this
    // receiver thread's entire body (called once, loops internally for the
    // thread's lifetime), so a stack array already has the right lifetime
    // and is inherently per-thread. `thread_local` here can hit glibc's
    // __tls_get_addr, which has a known race with concurrent dlopen() (e.g.
    // CUDA driver/NVRTC/cuSOLVER lazy-loading during pipeline construction)
    // -- intermittent SIGSEGV in __tls_get_addr on a freshly started thread.
    struct mmsghdr msgs[BATCH_SIZE];
    struct iovec iovecs[BATCH_SIZE];
    struct sockaddr_in client_addrs[BATCH_SIZE];
    // Control buffer for SO_RXQ_OVFL ancillary data.  Only the last message
    // in each batch needs it — the counter is cumulative on the socket, so
    // reading it once per recvmmsg call is sufficient.
    alignas(struct cmsghdr) char ctrl_buf[CMSG_SPACE(sizeof(uint32_t))];

    const size_t slot_cap = state.slot_data_capacity();
    memset(msgs, 0, sizeof(msgs));
    for (int i = 0; i < BATCH_SIZE; ++i) {
      msgs[i].msg_hdr.msg_iov = &iovecs[i];
      msgs[i].msg_hdr.msg_iovlen = 1;
      msgs[i].msg_hdr.msg_name = &client_addrs[i];
      msgs[i].msg_hdr.msg_namelen = sizeof(client_addrs[i]);
    }

    // Opt-in local-staging path (SPATIAL_CAPTURE_STAGE_LOCAL): on this box the
    // ring buffer is first-touch-allocated on the GPU's NUMA node (see
    // ProcessorState's constructor), but two of the three live NICs are on
    // the *other* socket -- every recvmmsg into a ring slot directly is a
    // small, kernel-mediated cross-socket copy_to_user for those threads'
    // capture. Staging into a buffer this thread first-touches itself (so
    // it lands node-local, same node this thread is pinned to via
    // SPATIAL_CAPTURE_CPUS) keeps recvmmsg's own copy local/cheap, and moves
    // the unavoidable cross-socket traffic into one bulk streaming-store
    // copy per batch instead of many small per-datagram ones. Off by
    // default: it's an extra copy, and whether bulking the cross-socket
    // traffic actually wins depends on the interconnect/traffic pattern --
    // measure before enabling.
    const bool use_local_stage =
        std::getenv("SPATIAL_CAPTURE_STAGE_LOCAL") != nullptr;
    std::vector<char> local_stage;
    if (use_local_stage) {
      // First touch (the zero-init below) happens on this thread, so the
      // pages land on whatever NUMA node this thread is currently pinned to.
      local_stage.assign(static_cast<size_t>(BATCH_SIZE) * slot_cap, 0);
      INFO_LOG("Capture thread {}: local-staging enabled ({} KB)", thread_id_,
               local_stage.size() / 1024);
    }

    std::cout << "Receiver thread started for ifname " << ifname << std::endl;

    void *slot_ptrs[BATCH_SIZE];
    int slot_indices[BATCH_SIZE];
    int lens_buf[BATCH_SIZE];
    sockaddr_in addr_buf[BATCH_SIZE];

    // Adaptive batch: tracks the recent drain size so light traffic doesn't
    // burn (and immediately abandon) 256 ring slots per datagram, while
    // sustained load quickly ramps back to full batches.
    int reserve_target = BATCH_SIZE;

    while (state.running) {
      // Wait for data before claiming any slots.  100 ms timeout keeps the
      // shutdown check responsive.
      struct pollfd pfd = {sockfd, POLLIN, 0};
      const int pret = poll(&pfd, 1, 100);
      if (pret <= 0) {
        if (pret < 0 && errno != EINTR && errno != EAGAIN) {
          std::cerr << "poll error on ifname " << ifname
                    << ": " << strerror(errno)
                    << " (errno=" << errno << ") revents=" << pfd.revents
                    << " — receiver thread exiting\n";
          ERROR_LOG("poll error on ifname {}: {} (errno={}) — receiver thread exiting",
                    ifname, strerror(errno), errno);
          break;
        }
        continue;
      }

      // Lock-free strided reservation: each capture thread claims its own
      // non-overlapping slots (thread_id_, thread_id_+N, ...), so no mutex
      // is needed and there is no ring-full spin inside a critical section.
      const int reserved =
          state.reserve_write_batch_strided(thread_id_, my_linear,
                                            reserve_target, slot_ptrs,
                                            slot_indices);
      for (int i = 0; i < reserved; ++i) {
        iovecs[i].iov_base = use_local_stage
                                 ? static_cast<void *>(local_stage.data() +
                                                        i * slot_cap)
                                 : slot_ptrs[i];
        iovecs[i].iov_len = slot_cap;
        msgs[i].msg_hdr.msg_namelen = sizeof(client_addrs[i]);
        msgs[i].msg_hdr.msg_control = nullptr;
        msgs[i].msg_hdr.msg_controllen = 0;
      }
      // Attach control buffer to last slot to receive SO_RXQ_OVFL counter.
      if (reserved > 0) {
        msgs[reserved - 1].msg_hdr.msg_control = ctrl_buf;
        msgs[reserved - 1].msg_hdr.msg_controllen = sizeof(ctrl_buf);
      }

      int ret_val = recvmmsg(sockfd, msgs, reserved, MSG_DONTWAIT, nullptr);
      if (ret_val < 0) {
        state.abandon_write_batch(reserved, slot_indices);
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
          continue;
        // Fatal receive error: without this log the thread dies silently
        // and the only visible symptom is packets_received freezing.
        ERROR_LOG("recvmmsg failed on ifname {}: {} — receiver thread exiting",
                  ifname, strerror(errno));
        std::cerr << "recvmmsg failed on ifname " << ifname << ": "
                  << strerror(errno) << " — receiver thread exiting\n";
        break;
      }

      // Read cumulative drop counter from SO_RXQ_OVFL cmsg on the last slot.
      // Works for both kernel sockets and VMA (which intercepts SO_RXQ_OVFL).
      const int cmsg_slot = ret_val > 0 ? ret_val - 1 : reserved - 1;
      struct msghdr *cmsg_hdr = &msgs[cmsg_slot].msg_hdr;
      for (struct cmsghdr *cm = CMSG_FIRSTHDR(cmsg_hdr); cm;
           cm = CMSG_NXTHDR(cmsg_hdr, cm)) {
        if (cm->cmsg_level == SOL_SOCKET && cm->cmsg_type == SO_RXQ_OVFL) {
          kernel_drops.store(*reinterpret_cast<uint32_t *>(CMSG_DATA(cm)),
                             std::memory_order_relaxed);
          break;
        }
      }

      // Bulk-copy the batch out of the local staging buffer into the real
      // (possibly remote-NUMA) ring slots. One streaming-store pass per
      // batch instead of recvmmsg's own per-datagram cross-socket copy.
      if (use_local_stage) {
        for (int i = 0; i < ret_val; ++i) {
          copy_nt(slot_ptrs[i], local_stage.data() + i * slot_cap,
                  slot_cap);
        }
      }

      for (int i = 0; i < ret_val; ++i) {
        lens_buf[i] = msgs[i].msg_len;
        addr_buf[i] = client_addrs[i];
      }
      state.commit_write_batch(ret_val, slot_indices, lens_buf, addr_buf);
      if (ret_val < reserved) {
        state.abandon_write_batch(reserved - ret_val, slot_indices + ret_val);
      }
      state.packets_received += ret_val;
    }
    std::cout << "Receiver thread exiting for ifname " << ifname << std::endl;
  };

  // Cumulative drop counter from SO_RXQ_OVFL ancillary data (works for both
  // kernel sockets and VMA, which intercepts SO_RXQ_OVFL).
  std::atomic<uint32_t> kernel_drops{0};
  uint32_t get_drops() const override {
    return kernel_drops.load(std::memory_order_relaxed);
  }

private:
  int sockfd;
  struct sockaddr_in server_addr;
  int port;
  int buffer_size;
  int recv_buffer_size;
  std::string ifname;
  int thread_id_{0};   // index of this capture thread (0..nr_threads_-1)
  int nr_threads_{1};  // total number of concurrent capture threads
};

class PCAPPacketCapture : public PacketInput {
public:
  PCAPPacketCapture(const std::string &pcap_filename, bool loop = false,
                    uint64_t seq_jump_per_packet = 64);
  ~PCAPPacketCapture();

  void get_packets(ProcessorStateBase &state) override {
    INFO_LOG("PCAP reader thread started for file: {}", filename_);

    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t *handle = nullptr;
    struct pcap_pkthdr *header;
    const u_char *data;
    int res;

    // Statistics
    uint64_t total_packets = 0;
    uint64_t total_bytes = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_stats_time = start_time;

    // For sample number incrementing across loops
    uint64_t current_loop = 0;
    uint64_t sample_offset = 0;
    uint64_t min_sample_in_loop = UINT64_MAX;
    uint64_t max_sample_in_loop = 0;
    bool first_loop = true;

    if (loop_ && seq_jump_per_packet_ > 0) {
      INFO_LOG("Sample incrementing enabled with seq jump per packet: {}",
               seq_jump_per_packet_);
    }

    do {
      handle = pcap_open_offline(filename_.c_str(), errbuf);
      if (!handle) {
        ERROR_LOG("Failed to open PCAP file '{}': {}", filename_, errbuf);
        state.running = 0;
        return;
      }

      if (current_loop > 0) {
        INFO_LOG("Starting loop {} with sample offset {}", current_loop,
                 sample_offset);
      } else {
        INFO_LOG("Reading packets from PCAP file...");
      }

      while ((res = pcap_next_ex(handle, &header, &data)) >= 0 &&
             state.running) {
        // Get pointer to write location
        void *write_pointer = state.get_current_write_pointer();

        if (header->caplen <= header->len &&
            header->caplen <= state.slot_data_capacity()) {
          std::memcpy(write_pointer, data, header->caplen);

          // Extract and potentially modify the sample number
          // Structure: Ethernet (14) + IP (20) + UDP (8) + CustomHeader
          // CustomHeader layout: sample_count (8), fpga_id (4), freq_channel
          // (2), padding (8)
          const size_t CUSTOM_HEADER_OFFSET = 42; // Ethernet + IP + UDP

          if (header->caplen > CUSTOM_HEADER_OFFSET + sizeof(uint64_t)) {
            // sample_count is the first field in CustomHeader (uint64_t)
            uint64_t *sample_count_ptr = reinterpret_cast<uint64_t *>(
                static_cast<uint8_t *>(write_pointer) + CUSTOM_HEADER_OFFSET);

            uint64_t original_sample = *sample_count_ptr;

            // Track min/max for first loop to calculate range
            if (first_loop) {
              min_sample_in_loop =
                  std::min(min_sample_in_loop, original_sample);
              max_sample_in_loop =
                  std::max(max_sample_in_loop, original_sample);
            }

            // Apply offset for loops after the first
            if (!first_loop && sample_offset > 0) {
              *sample_count_ptr = original_sample + sample_offset;
              DEBUG_LOG("Modified sample count: {} -> {}", original_sample,
                        *sample_count_ptr);
            }
          }

          // Extract IP source address from the raw Ethernet frame so that
          // OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET works correctly in PCAP
          // replay mode. Layout: Ethernet (14) + 12 bytes into IP header =
          // offset 26 for the 4-byte source IP (network byte order).
          struct sockaddr_in client_addr;
          std::memset(&client_addr, 0, sizeof(client_addr));
          client_addr.sin_family = AF_INET;
          constexpr size_t IP_SRC_OFFSET = 26;
          if (header->caplen > IP_SRC_OFFSET + 4) {
            client_addr.sin_addr.s_addr =
                *reinterpret_cast<const uint32_t *>(
                    static_cast<const uint8_t *>(write_pointer) +
                    IP_SRC_OFFSET);
          } else {
            client_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
          }
          client_addr.sin_port = htons(0);

          // Add metadata
          state.add_received_packet_metadata(header->caplen, client_addr);
          state.packets_received += 1;

          total_packets++;
          total_bytes += header->caplen;

          // Get next write pointer for next iteration
          state.get_next_write_pointer();

          // Periodic statistics (every 5 seconds)
          auto now = std::chrono::steady_clock::now();
          auto stats_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                   now - last_stats_time)
                                   .count();

          if (stats_elapsed >= 5) {
            auto total_elapsed =
                std::chrono::duration_cast<std::chrono::seconds>(now -
                                                                 start_time)
                    .count();
            double avg_pps = total_packets / (double)total_elapsed;
            double avg_mbps = (total_bytes * 8.0) / (total_elapsed * 1e6);
            INFO_LOG("PCAP stats: {:.2f} kpps, {:.2f} Mbps (total: {} packets, "
                     "loop: {})",
                     avg_pps / 1000.0, avg_mbps, total_packets, current_loop);
            last_stats_time = now;
          }
        } else {
          WARN_LOG("Packet truncated: caplen={} < len={}", header->caplen,
                   header->len);
        }
      }

      if (res == -1) {
        ERROR_LOG("Error reading PCAP: {}", pcap_geterr(handle));
      } else {
        INFO_LOG("Reached end of PCAP file. Total packets read: {}",
                 total_packets);
      }

      pcap_close(handle);
      handle = nullptr;

      if (loop_ && state.running) {
        // Calculate offset for next loop
        if (first_loop) {
          if (max_sample_in_loop > min_sample_in_loop) {
            // Range of samples in this loop
            uint64_t sample_range = max_sample_in_loop - min_sample_in_loop;
            // Next loop starts at: max + seq_jump_per_packet
            sample_offset = sample_range + seq_jump_per_packet_;

            INFO_LOG("First loop complete. Sample range: {} to {} (span: {})",
                     min_sample_in_loop, max_sample_in_loop, sample_range);
            INFO_LOG("Next loop offset will be: {}", sample_offset);
          } else {
            WARN_LOG("Could not determine sample range, using "
                     "seq_jump_per_packet only");
            sample_offset = seq_jump_per_packet_;
          }
          first_loop = false;
        } else {
          // For subsequent loops, keep adding the same offset
          uint64_t sample_range = max_sample_in_loop - min_sample_in_loop;
          sample_offset += sample_range + seq_jump_per_packet_;
        }

        current_loop++;
        INFO_LOG(
            "Looping back to beginning (loop {}, cumulative offset: {})...",
            current_loop, sample_offset);

        // Small delay before restarting to avoid tight loop
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

    } while (loop_ && state.running);

    INFO_LOG("PCAP reader thread exiting. Total packets: {}, Total bytes: {}, "
             "Loops: {}",
             total_packets, total_bytes, current_loop);
  }

private:
  std::string filename_;
  bool loop_; // Whether to loop the PCAP file
  uint64_t
      seq_jump_per_packet_; // Sequence number jump between packets (e.g., 64)
};

class PCAPMultiFPGAPacketCapture : public PacketInput {
public:
  PCAPMultiFPGAPacketCapture(const std::string &pcap_filename,
                             bool loop = false, int num_fpgas = 4,
                             uint64_t seq_jump_per_packet = 64);
  ~PCAPMultiFPGAPacketCapture();

  void get_packets(ProcessorStateBase &state) override {
    INFO_LOG("PCAP reader thread started for file: {}", filename_);

    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t *handle = nullptr;
    struct pcap_pkthdr *header;
    const u_char *data;
    int res;

    // Statistics
    uint64_t total_packets = 0;
    uint64_t total_bytes = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_stats_time = start_time;

    // For sample number incrementing across loops
    uint64_t current_loop = 0;
    uint64_t sample_offset = 0;
    uint64_t min_sample_in_loop = UINT64_MAX;
    uint64_t max_sample_in_loop = 0;
    bool first_loop = true;

    if (loop_ && seq_jump_per_packet_ > 0) {
      INFO_LOG("Sample incrementing enabled with seq jump per packet: {}",
               seq_jump_per_packet_);
    }

    do {
      handle = pcap_open_offline(filename_.c_str(), errbuf);
      if (!handle) {
        ERROR_LOG("Failed to open PCAP file '{}': {}", filename_, errbuf);
        state.running = 0;
        return;
      }

      if (current_loop > 0) {
        INFO_LOG("Starting loop {} with sample offset {}", current_loop,
                 sample_offset);
      } else {
        INFO_LOG("Reading packets from PCAP file...");
      }

      while ((res = pcap_next_ex(handle, &header, &data)) >= 0 &&
             state.running) {
        // Get pointer to write location
        for (int i = 0; i < num_fpgas; ++i) {
          void *write_pointer = state.get_current_write_pointer();

          if (header->caplen <= header->len &&
              header->caplen <= state.slot_data_capacity()) {
            std::memcpy(write_pointer, data, header->caplen);

            // Extract and potentially modify the sample number
            // Structure: Ethernet (14) + IP (20) + UDP (8) + CustomHeader
            // CustomHeader layout: sample_count (8), fpga_id (4), freq_channel
            // (2), padding (8)
            const size_t CUSTOM_HEADER_OFFSET = 42; // Ethernet + IP + UDP

            if (header->caplen > CUSTOM_HEADER_OFFSET + sizeof(uint64_t)) {
              // sample_count is the first field in CustomHeader (uint64_t)
              uint8_t *custom_header_start =
                  static_cast<uint8_t *>(write_pointer) + CUSTOM_HEADER_OFFSET;
              uint64_t *sample_count_ptr =
                  reinterpret_cast<uint64_t *>(custom_header_start);

              uint32_t *fpga_id_ptr =
                  reinterpret_cast<uint32_t *>(custom_header_start + 8);

              *fpga_id_ptr = i;
              uint64_t original_sample = *sample_count_ptr;

              // Track min/max for first loop to calculate range
              if (first_loop) {
                min_sample_in_loop =
                    std::min(min_sample_in_loop, original_sample);
                max_sample_in_loop =
                    std::max(max_sample_in_loop, original_sample);
              }

              // Apply offset for loops after the first
              if (!first_loop && sample_offset > 0) {
                *sample_count_ptr = original_sample + sample_offset;
                DEBUG_LOG("Modified sample count: {} -> {}", original_sample,
                          *sample_count_ptr);
              }
            }

            // Extract IP source address from the raw Ethernet frame so that
            // OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET works correctly in PCAP
            // replay mode. Layout: Ethernet (14) + 12 bytes into IP header =
            // offset 26 for the 4-byte source IP (network byte order).
            struct sockaddr_in client_addr;
            std::memset(&client_addr, 0, sizeof(client_addr));
            client_addr.sin_family = AF_INET;
            constexpr size_t IP_SRC_OFFSET = 26;
            if (header->caplen > IP_SRC_OFFSET + 4) {
              client_addr.sin_addr.s_addr =
                  *reinterpret_cast<const uint32_t *>(
                      static_cast<const uint8_t *>(write_pointer) +
                      IP_SRC_OFFSET);
            } else {
              client_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            }
            client_addr.sin_port = htons(0);

            // Add metadata
            state.add_received_packet_metadata(header->caplen, client_addr);
            state.packets_received += 1;

            total_packets++;
            total_bytes += header->caplen;

            // Get next write pointer for next iteration
            state.get_next_write_pointer();
          } else {
            WARN_LOG("Packet truncated: caplen={} < len={}", header->caplen,
                     header->len);
          }
        }
        // Periodic statistics (every 5 seconds)
        auto now = std::chrono::steady_clock::now();
        auto stats_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                 now - last_stats_time)
                                 .count();

        if (stats_elapsed >= 5) {
          auto total_elapsed =
              std::chrono::duration_cast<std::chrono::seconds>(now - start_time)
                  .count();
          double avg_pps = total_packets / (double)total_elapsed;
          double avg_mbps = (total_bytes * 8.0) / (total_elapsed * 1e6);
          INFO_LOG("PCAP stats: {:.2f} kpps, {:.2f} Mbps (total: {} packets, "
                   "loop: {})",
                   avg_pps / 1000.0, avg_mbps, total_packets, current_loop);
          last_stats_time = now;
        }
      }

      if (res == -1) {
        ERROR_LOG("Error reading PCAP: {}", pcap_geterr(handle));
      } else {
        INFO_LOG("Reached end of PCAP file. Total packets read: {}",
                 total_packets);
      }

      pcap_close(handle);
      handle = nullptr;

      if (loop_ && state.running) {
        // Calculate offset for next loop
        if (first_loop) {
          if (max_sample_in_loop > min_sample_in_loop) {
            // Range of samples in this loop
            uint64_t sample_range = max_sample_in_loop - min_sample_in_loop;
            // Next loop starts at: max + seq_jump_per_packet
            sample_offset = sample_range + seq_jump_per_packet_;

            INFO_LOG("First loop complete. Sample range: {} to {} (span: {})",
                     min_sample_in_loop, max_sample_in_loop, sample_range);
            INFO_LOG("Next loop offset will be: {}", sample_offset);
          } else {
            WARN_LOG("Could not determine sample range, using "
                     "seq_jump_per_packet only");
            sample_offset = seq_jump_per_packet_;
          }
          first_loop = false;
        } else {
          // For subsequent loops, keep adding the same offset
          uint64_t sample_range = max_sample_in_loop - min_sample_in_loop;
          sample_offset += sample_range + seq_jump_per_packet_;
        }

        current_loop++;
        INFO_LOG(
            "Looping back to beginning (loop {}, cumulative offset: {})...",
            current_loop, sample_offset);

        // Small delay before restarting to avoid tight loop
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    } while (loop_ && state.running);

    INFO_LOG("PCAP reader thread exiting. Total packets: {}, Total bytes: {}, "
             "Loops: {}",
             total_packets, total_bytes, current_loop);
  }

private:
  std::string filename_;
  bool loop_; // Whether to loop the PCAP file
  uint64_t
      seq_jump_per_packet_; // Sequence number jump between packets (e.g., 64)
  int num_fpgas;
};
