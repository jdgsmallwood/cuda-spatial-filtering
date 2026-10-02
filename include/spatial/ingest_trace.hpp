#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace spatial {

inline int completion_poll_interval_us(const char *value) {
  if (!value) return 200;
  char *end = nullptr;
  const long interval = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || interval < 50 || interval > 10000)
    throw std::invalid_argument(
        "SPATIAL_COMPLETION_POLL_US must be an integer from 50 to 10000");
  return static_cast<int>(interval);
}

inline int coordinator_cpu_from_env(const char *value) {
  if (!value) return -1;
  char *end = nullptr;
  const long cpu = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || cpu < 0 || cpu >= CPU_SETSIZE)
    throw std::invalid_argument("SPATIAL_COORDINATOR_CPU must be a valid CPU index");
  return static_cast<int>(cpu);
}

// Each lane has exactly one producer. Read/export only after all producers
// have joined. Allocation and first-touch happen before capture starts;
// recording takes no locks and performs no file I/O. Bounded rings retain
// the most recent events and report overwritten counts on export.
class IngestTrace {
public:
  enum Event : uint32_t {
    CompletionCycle = 1, WorkerPause, WorkerBatch, ReceiveRingWait,
    ReceiveProgressGap, ReceiveBatch, PipelineSubmit, BufferSeal
  };
  static constexpr size_t LaneCount = 10;
  struct Record {
    uint64_t start_ns, end_ns, arg0, arg1, arg2;
    uint32_t event;
    int cpu;
  };
  struct alignas(64) Lane {
    std::vector<Record> records;
    uint64_t written = 0;
    long tid = 0;
  };

  explicit IngestTrace(const char *path = std::getenv("SPATIAL_INGEST_TRACE_FILE"),
                       size_t capacity = 131072)
      : path_(path ? path : "") {
    if (path_.empty()) return;
    if (!capacity) throw std::invalid_argument("trace capacity must be positive");
    for (size_t i = 0; i < LaneCount; ++i)
      lanes_[i].records.resize(capacity * (i == 0 ? 4 : 1));
  }
  bool enabled() const { return !path_.empty(); }
  static uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
  }
  static uint64_t thread_cpu_ns() {
    timespec value{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value);
    return static_cast<uint64_t>(value.tv_sec) * 1000000000 + value.tv_nsec;
  }
  void record(size_t lane, Event event, uint64_t start, uint64_t end,
              uint64_t arg0 = 0, uint64_t arg1 = 0, uint64_t arg2 = 0) {
    if (!enabled()) return;
    auto &out = lanes_.at(lane);
    if (!out.tid) out.tid = syscall(SYS_gettid);
    out.records[out.written++ % out.records.size()] =
        Record{start, end, arg0, arg1, arg2, event, sched_getcpu()};
  }
  const Lane &lane(size_t index) const { return lanes_.at(index); }
  void dump() const {
    if (!enabled()) return;
    std::ofstream out(path_);
    if (!out) throw std::runtime_error("cannot write ingest trace: " + path_);
    out << "# monotonic timestamps in ns; CPU is sampled at interval end\n"
        << "# lanes: 0=completion, 1..4=workers, 5..8=receivers, 9=feeder\n"
        << "# events: 1=completion (arg0=lock_acquired_ns,arg1=missing_delta,arg2=thread_cpu_ns), "
           "2=worker_pause, 3=worker_batch (arg0=packets), 4=ring_wait, "
           "5=receive_progress_gap, 6=receive_batch (arg0=CQEs), 7=pipeline_submit, "
           "8=buffer_seal (arg0=window; workers continue on other buffers)\n";
    for (size_t i = 0; i < LaneCount; ++i) {
      const auto &l = lanes_[i];
      out << "# lane=" << i << " tid=" << l.tid << " written=" << l.written
          << " overwritten=" << (l.written > l.records.size()
                                    ? l.written - l.records.size() : 0) << '\n';
    }
    out << "lane,tid,cpu,event,start_ns,end_ns,arg0,arg1,arg2\n";
    for (size_t i = 0; i < LaneCount; ++i) {
      const auto &l = lanes_[i];
      const uint64_t first = l.written > l.records.size()
                                 ? l.written - l.records.size() : 0;
      for (uint64_t j = first; j < l.written; ++j) {
        const auto &r = l.records[j % l.records.size()];
        out << i << ',' << l.tid << ',' << r.cpu << ',' << r.event << ','
            << r.start_ns << ',' << r.end_ns << ',' << r.arg0 << ',' << r.arg1
            << ',' << r.arg2 << '\n';
      }
    }
    if (!out) throw std::runtime_error("incomplete ingest trace write: " + path_);
  }
private:
  std::string path_;
  std::array<Lane, LaneCount> lanes_;
};
} // namespace spatial
