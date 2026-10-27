#pragma once

#include "spatial/packet_formats.hpp"
#include "spatial/ingest_trace.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace spatial {

// Experimental CPU-ring ingest. A slot has an immutable logical window number
// for its entire lifetime. Workers cache the atomic descriptor once per batch;
// CLOSING waits for each worker's release acknowledgement before touching data.
// No worker waits for the coordinator or acquires a completion mutex.
template <typename T, size_t N, int W> class DecoupledIngest {
  static_assert(N >= 2, "overlap/lookahead needs at least two host buffers");
  static_assert(W == T::NR_FPGA_SOURCES, "one worker per FPGA/NIC is required");
  enum Phase : uint64_t { Open = 0, Closing = 1, InFlight = 2 };
  static uint64_t token(uint64_t window, Phase phase) {
    return ((window + 1) << 2) | phase;
  }
  static uint64_t window_of(uint64_t word) { return (word >> 2) - 1; }
  static Phase phase_of(uint64_t word) { return static_cast<Phase>(word & 3); }
  struct alignas(64) Slot { std::atomic<uint64_t> word{0}; };
  struct Pending {
    int index;
    uint64_t sample;
    int channel;
    bool operator<(const Pending &other) const { return sample > other.sample; }
  };
  struct alignas(64) Worker {
    std::array<std::atomic<uint64_t>, N> acknowledged{};
    std::array<uint64_t, N> view{};
    std::array<uint64_t, T::NR_FPGA_CHANNELS> latest{};
    std::vector<Pending> pending;
    std::chrono::steady_clock::time_point stuck_since{};
    bool stuck = false;
  };
  struct Destination { uint64_t window; int packet; };
  using Parsed = ProcessedPacket<typename T::PacketScaleStructure,
                                 typename T::PacketDataStructure>;
  std::array<Slot, N> slots_;
  std::array<Worker, W> workers_;
  std::array<uint64_t, T::NR_FPGA_SOURCES> origin_{};
  uint64_t next_close_ = 0; // coordinator only
  uint64_t close_start_ns_ = 0;

  template <typename S>
  void record_batch(S &s, int worker, uint64_t start, uint64_t count) {
#ifdef SPATIAL_TIMING_DIAGNOSTICS
    if (!start) return;
    const uint64_t end = S::timing_now_ns();
    s.timing_worker_batch[worker].record(end - start);
#ifdef SPATIAL_INGEST_TRACE
    if (end - start >= 200000)
      s.ingest_trace.record(1 + worker, IngestTrace::WorkerBatch, start, end, count, count);
#endif
#endif
  }

  template <typename S>
  int destinations(const S &s, uint64_t sample, int fp,
                   std::array<Destination, 2> &out) const {
    // Match the established signed packet-index arithmetic, including guards.
    const int64_t index = (static_cast<int64_t>(sample) -
        static_cast<int64_t>(origin_[fp])) / static_cast<int64_t>(s.NR_BETWEEN_SAMPLES);
    if (index < -1) return 0;
    if (index == -1) { out[0] = {0, 0}; return 1; }
    const uint64_t window = static_cast<uint64_t>(index) / s.NR_PACKETS_FOR_CORRELATION;
    const int packet = static_cast<int>(index % s.NR_PACKETS_FOR_CORRELATION);
    out[0] = {window, packet + 1};
    if (packet == 0 && window > 0) {
      out[1] = {window - 1, static_cast<int>(s.NR_PACKETS_FOR_CORRELATION) + 1};
      return 2;
    }
    if (packet == static_cast<int>(s.NR_PACKETS_FOR_CORRELATION) - 1) {
      out[1] = {window + 1, 0};
      return 2;
    }
    return 1;
  }

  void refresh(int worker) {
    for (size_t b = 0; b < N; ++b)
      workers_[worker].view[b] = slots_[b].word.load(std::memory_order_acquire);
  }

  // Closing remains writable by this worker until its acknowledgement. That
  // lets it drain already-deferred packets for the closing generation first.
  bool writable(const Worker &w, const Destination &d) const {
    const size_t b = d.window % N;
    const uint64_t word = w.view[b];
    return word && window_of(word) == d.window &&
        (phase_of(word) == Open ||
         (phase_of(word) == Closing &&
          w.acknowledged[b].load(std::memory_order_relaxed) != d.window + 1));
  }

  template <typename S>
  bool place(S &s, int worker, Parsed &pkt) {
    auto &w = workers_[worker];
    std::array<Destination, 2> dest{};
    const int count = destinations(s, pkt.sample_count, worker, dest);
    // Keep the whole packet until every future destination exists. This also
    // preserves the leading guard when the next window is waiting for H2D.
    for (int i = 0; i < count; ++i) {
      const uint64_t word = w.view[dest[i].window % N];
      if (!word || window_of(word) < dest[i].window) return false;
    }
    const int ch = pkt.freq_channel - s.MIN_FREQ_CHANNEL;
    bool copied = false;
    for (int i = 0; i < count; ++i) {
      const auto &d = dest[i];
      if (!writable(w, d)) continue;
      auto *data = s.d_samples[d.window % N];
      auto &samples = (*data->samples)[ch][d.packet][worker];
      auto &scales = (*data->scales)[ch][d.packet][worker * T::NR_RECEIVERS_PER_PACKET];
      // Definition supplied by spatial.hpp before this template is instantiated.
      copy_nt(&samples, pkt.sample_data(), sizeof(typename T::PacketDataStructure), true);
      pkt.copy_scales(&scales);
      data->arrivals[0][ch][d.packet][worker] = true;
      copied = true;
    }
    if (!copied) s.packets_discarded.fetch_add(1, std::memory_order_relaxed);
    w.latest[ch] = std::max(w.latest[ch], pkt.sample_count);
    pkt.original_packet_processed->store(true, std::memory_order_release);
    return true;
  }

  template <typename S>
  bool parse(S &s, int worker, int index, Parsed &pkt) {
    auto *entry = s.d_packet_data[index];
    if (entry->processed.load(std::memory_order_relaxed)) return false;
    using Entry = typename T::PacketEntryType;
    pkt = entry->Entry::parse(s.packet_format);
    const int ch = static_cast<int>(pkt.freq_channel) - static_cast<int>(s.MIN_FREQ_CHANNEL);
    if (!pkt.valid() || ch < 0 || ch >= static_cast<int>(T::NR_FPGA_CHANNELS)) {
      entry->processed.store(true, std::memory_order_release);
      s.packets_discarded.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    int fp = pkt.fpga_id < s.fpga_index_lut.size()
        ? s.fpga_index_lut[pkt.fpga_id] : -1;
    if (fp < 0) {
      const auto it = s.fpga_ids.find(pkt.fpga_id);
      if (it != s.fpga_ids.end()) fp = it->second;
    }
    if (fp != worker || (pkt.sample_count == 0 && !pkt.codif_header) ||
        pkt.sample_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      std::cerr << "Decoupled ingest requires one configured FPGA per NIC worker; invalid packet on worker "
                << worker << '\n';
      s.shutdown();
      return false;
    }
    if (!s.buffer_init_flag.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lock(s.buffer_bootstrap_mutex);
      if (!s.buffer_init_flag.load(std::memory_order_relaxed)) {
        s.initialize_buffers(pkt.sample_count, pkt.fpga_id);
        s.buffer_init_flag.store(true, std::memory_order_release);
      }
      refresh(worker);
    }
    return true;
  }

  template <typename S>
  void publish_and_acknowledge(S &s, int worker) {
    auto &w = workers_[worker];
    // Covers normal batches, deferred copies, and the idle acknowledgement path.
    // A release store alone does not order x86 streaming stores.
    _mm_sfence();
    for (size_t ch = 0; ch < T::NR_FPGA_CHANNELS; ++ch) {
      if (!w.latest[ch]) continue;
      S::atomic_max_u64(s.latest_packet_received[ch][worker], w.latest[ch]);
      w.latest[ch] = 0;
    }
    for (size_t b = 0; b < N; ++b) {
      const uint64_t word = w.view[b];
      if (!word || phase_of(word) != Closing) continue;
      const uint64_t window = window_of(word);
      if (w.acknowledged[b].load(std::memory_order_relaxed) == window + 1) continue;
      bool queued_for_window = false;
      for (const auto &p : w.pending) {
        std::array<Destination, 2> dest{};
        const int count = destinations(s, p.sample, worker, dest);
        for (int i = 0; i < count; ++i)
          queued_for_window |= dest[i].window == window;
      }
      if (!queued_for_window)
        w.acknowledged[b].store(window + 1, std::memory_order_release);
    }
  }

public:
  explicit DecoupledIngest(size_t ring_slots_per_worker) {
    for (auto &worker : workers_) worker.pending.reserve(ring_slots_per_worker);
  }

  template <typename S> void initialize(S &s) {
    origin_ = s.buffers[0].start_seq;
    for (size_t b = 0; b < N; ++b)
      slots_[b].word.store(token(b, Open), std::memory_order_release);
  }

  template <typename S> void release(S &s, int buffer) {
    const uint64_t word = slots_[buffer].word.load(std::memory_order_acquire);
    if (!word || phase_of(word) != InFlight)
      throw std::logic_error("releasing a buffer that is not owned by H2D");
    std::memset(s.d_samples[buffer]->arrivals, 0, sizeof(typename T::PacketFinalDataType::ArrivalsType));
    // Only this slot's generation changes. It is never reassigned according
    // to another slot's release order, including out-of-order GPU callbacks.
    slots_[buffer].word.store(token(window_of(word) + N, Open), std::memory_order_release);
#ifdef SPATIAL_DIAGNOSTICS
    s.diagnostic_buffer_releases.fetch_add(1, std::memory_order_relaxed);
#endif
  }

  template <typename S> void coordinator_step(S &s) {
    if (!s.buffer_init_flag.load(std::memory_order_acquire)) return;
    const size_t b = next_close_ % N;
    uint64_t word = slots_[b].word.load(std::memory_order_acquire);
    if (!word || window_of(word) != next_close_) return;
    const uint64_t last_offset = ((next_close_ + 1) * s.NR_PACKETS_FOR_CORRELATION - 1) * s.NR_BETWEEN_SAMPLES;
    if (phase_of(word) == Open) {
      for (size_t ch = 0; ch < T::NR_FPGA_CHANNELS; ++ch)
        for (int fp = 0; fp < W; ++fp)
          if (s.latest_packet_received[ch][fp].load(std::memory_order_acquire) <
              origin_[fp] + last_offset + s.NR_BETWEEN_SAMPLES / 2)
            return;
      close_start_ns_ = IngestTrace::now_ns();
      slots_[b].word.store(token(next_close_, Closing), std::memory_order_release);
      return; // never wait for a worker, nor stop one from copying other slots
    }
    if (phase_of(word) != Closing) return;
    for (int worker = 0; worker < W; ++worker)
      if (workers_[worker].acknowledged[b].load(std::memory_order_acquire) != next_close_ + 1)
        return;
    slots_[b].word.store(token(next_close_, InFlight), std::memory_order_release);
#ifdef SPATIAL_INGEST_TRACE
    s.ingest_trace.record(0, IngestTrace::BufferSeal, close_start_ns_, IngestTrace::now_ns(), next_close_);
#endif
    auto *data = s.d_samples[b];
    data->start_seq_id = origin_[0] + next_close_ * s.NR_PACKETS_FOR_CORRELATION * s.NR_BETWEEN_SAMPLES;
    data->end_seq_id = origin_[0] + last_offset;
#ifdef SPATIAL_TIMING_DIAGNOSTICS
    const uint64_t finalize_start = S::timing_now_ns();
    s.timing_handoff_start_ns[b].store(finalize_start, std::memory_order_relaxed);
#endif
    data->zero_missing_packets();
    s.packets_missing += data->get_num_missing_packets();
#ifdef SPATIAL_DIAGNOSTICS
    s.diagnostic_buffer_handoffs.fetch_add(1, std::memory_order_relaxed);
    for (size_t ch = 0; ch < T::NR_FPGA_CHANNELS; ++ch)
      for (size_t p = 1; p <= s.NR_PACKETS_FOR_CORRELATION; ++p)
        for (int fp = 0; fp < W; ++fp)
          if (!data->arrivals[0][ch][p][fp])
            s.diagnostic_missing_by_fpga[fp].fetch_add(1, std::memory_order_relaxed);
#endif
    ++next_close_;
    if (s.synchronous_pipeline) {
      s.pipeline_->execute_pipeline(data);
      ++s.pipeline_runs_queued;
    } else {
      static_assert(N < S::PIPELINE_Q_CAP, "handoff queue must hold every host buffer");
      const uint64_t tail = s.pipeline_q_tail_.v.load(std::memory_order_relaxed);
      s.pipeline_q_buf_[tail % S::PIPELINE_Q_CAP] = b;
      s.pipeline_q_tail_.v.store(tail + 1, std::memory_order_release);
    }
#ifdef SPATIAL_TIMING_DIAGNOSTICS
    s.timing_buffer_handoff.record(S::timing_now_ns() - finalize_start);
#endif
  }

  template <typename S> void run_worker(S &s, int worker) {
    static_assert(S::DECOUPLED_RING_SIZE % W == 0);
    static_assert(S::DECOUPLED_RING_SIZE / W > 1);
    auto &w = workers_[worker];
    uint64_t read = worker;
    const int stride = W;
    // Preserve the matched A/B binary's accumulation policy.
#ifdef SPATIAL_DIRECT_BATCH_TARGET
    constexpr int batch_target = SPATIAL_DIRECT_BATCH_TARGET;
#else
    constexpr int batch_target = 192;
#endif
#ifdef SPATIAL_DIRECT_BATCH_WAIT_US
    constexpr int batch_wait_us = SPATIAL_DIRECT_BATCH_WAIT_US;
#else
    constexpr int batch_wait_us = 200;
#endif
    while (s.running.load(std::memory_order_acquire)) {
      if (s.buffer_init_flag.load(std::memory_order_acquire)) refresh(worker);
      uint64_t processed = 0;
      size_t drained = 0;
      uint64_t drain_start = 0;
#ifdef SPATIAL_TIMING_DIAGNOSTICS
      if (!w.pending.empty()) drain_start = S::timing_now_ns();
#endif
      // Deferred entries stay in their NIC's ring until this same worker
      // releases them. Ordered draining cannot publish a watermark past an
      // older, deferred packet merely because a later slot was recycled first.
      while (!w.pending.empty() && drained < 256) {
        const auto pending = w.pending.front();
        Parsed pkt{};
        if (parse(s, worker, pending.index, pkt) && !place(s, worker, pkt)) break;
        std::pop_heap(w.pending.begin(), w.pending.end());
        w.pending.pop_back();
        ++processed;
        ++drained;
      }
      if (drained == 256) {
        publish_and_acknowledge(s, worker);
        record_batch(s, worker, drain_start, drained);
        s.packets_processed.fetch_add(processed, std::memory_order_release);
        continue;
      }
      if (drained) {
        publish_and_acknowledge(s, worker);
        record_batch(s, worker, drain_start, drained);
      }
      const uint64_t claim = s.per_thread_claim[worker].linear.load(std::memory_order_acquire);
      const int start = static_cast<int>(read % S::DECOUPLED_RING_SIZE);
      int available = static_cast<int>(std::min<uint64_t>((claim > read ? claim - read : 0) / stride,
          std::min<size_t>(256, S::DECOUPLED_RING_SIZE / stride - 1)));
      int committed = 0;
      int index = start;
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(batch_wait_us);
      do {
        while (committed < available && s.d_packet_data[index]->committed.load(std::memory_order_acquire)) {
          ++committed;
          index = (index + stride) % S::DECOUPLED_RING_SIZE;
        }
        if (committed >= batch_target || !committed || std::chrono::steady_clock::now() >= deadline) break;
        const uint64_t next_claim = s.per_thread_claim[worker].linear.load(std::memory_order_acquire);
        available = static_cast<int>(std::min<uint64_t>(
            (next_claim > read ? next_claim - read : 0) / stride,
            std::min<size_t>(256, S::DECOUPLED_RING_SIZE / stride - 1)));
        _mm_pause();
      } while (s.running.load(std::memory_order_acquire));
      uint64_t batch_start = 0;
#ifdef SPATIAL_TIMING_DIAGNOSTICS
      if (committed) batch_start = S::timing_now_ns();
#endif
      index = start;
      for (int i = 0; i < committed; ++i) {
        Parsed pkt{};
        if (s.d_packet_data[index]->length > 0 && parse(s, worker, index, pkt)) {
          if (w.pending.empty() && place(s, worker, pkt)) ++processed;
          else {
            w.pending.push_back({index, pkt.sample_count, static_cast<int>(pkt.freq_channel - s.MIN_FREQ_CHANNEL)});
            std::push_heap(w.pending.begin(), w.pending.end());
            s.packets_future_queued.fetch_add(1, std::memory_order_relaxed);
          }
        }
        index = (index + stride) % S::DECOUPLED_RING_SIZE;
      }
      if (!w.pending.empty()) {
        const auto now = std::chrono::steady_clock::now();
        if (!w.stuck || drained) { w.stuck = true; w.stuck_since = now; }
        else if (now - w.stuck_since >= s.future_stall_force_threshold) {
          // The established bounded-skew timeout is retained for real stream
          // jumps. A closing generation is not acknowledged until any queued
          // packets that belong to it have actually been copied.
          for (const auto &p : w.pending)
            w.latest[p.channel] = std::max(w.latest[p.channel], p.sample);
          w.stuck_since = now;
        }
      } else w.stuck = false;
      publish_and_acknowledge(s, worker);
      record_batch(s, worker, batch_start, committed);
      if (processed) s.packets_processed.fetch_add(processed, std::memory_order_release);
      read += static_cast<uint64_t>(committed) * stride;
      s.per_thread_read_linear[worker].linear.store(read, std::memory_order_release);
#ifdef SPATIAL_DIAGNOSTICS
      if (committed) s.diagnostic_processor_dispatches.fetch_add(1, std::memory_order_relaxed);
#endif
      if (!committed && !drained) _mm_pause();
    }
    s.workers_alive.fetch_sub(1, std::memory_order_acq_rel);
  }
};
} // namespace spatial
