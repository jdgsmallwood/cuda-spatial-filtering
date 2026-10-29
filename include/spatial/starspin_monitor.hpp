#pragma once

#include "spatial/redis_monitor.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <complex>
#include <fstream>
#include <map>
#include <sstream>
#include <unistd.h>
#include <time.h>

namespace spatial_monitor {
using Json = nlohmann::json;
inline int64_t wall_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
inline std::string read_text(const std::string &path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
using CpuTicks = std::array<uint64_t, 8>;
inline std::map<std::string, CpuTicks> parse_cpu(const std::string &text) {
  std::map<std::string, CpuTicks> result;
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    std::istringstream row(line);
    std::string name;
    row >> name;
    if (name.compare(0, 3, "cpu") != 0) continue;
    CpuTicks ticks{};
    bool valid = true;
    for (auto &tick : ticks) if (!(row >> tick)) valid = false;
    if (valid) result[name] = ticks; // guest time is already included in user/nice.
  }
  return result;
}
inline Json cpu_delta(const CpuTicks &before, const CpuTicks &after) {
  CpuTicks delta{};
  double total = 0;
  for (size_t i = 0; i < delta.size(); ++i) {
    if (after[i] < before[i]) return Json::object();
    delta[i] = after[i] - before[i];
    total += delta[i];
  }
  if (total == 0) return Json::object();
  return {{"user", 100.0 * (delta[0] + delta[1]) / total},
          {"system", 100.0 * (delta[2] + delta[5] + delta[6]) / total},
          {"idle", 100.0 * delta[3] / total},
          {"iowait", 100.0 * delta[4] / total},
          {"busy", 100.0 * (total - delta[3] - delta[4]) / total}};
}
inline std::map<std::string, uint64_t> parse_kib(const std::string &text) {
  std::map<std::string, uint64_t> result;
  std::istringstream lines(text);
  std::string line, name, unit;
  uint64_t value;
  while (std::getline(lines, line)) {
    std::istringstream row(line);
    if (row >> name >> value >> unit && unit == "kB")
      result[name.substr(0, name.size() - 1)] = value * 1024;
  }
  return result;
}
class HostSampler {
public:
  Json sample() {
    Json values = Json::object();
    auto cpus = parse_cpu(read_text("/proc/stat"));
    for (const auto &[name, ticks] : cpus) {
      auto prior = previous_.find(name);
      if (prior == previous_.end()) continue;
      const auto delta = cpu_delta(prior->second, ticks);
      for (const auto &[kind, value] : delta.items())
        values[name + "_" + kind + "_percent"] = value;
    }
    previous_ = std::move(cpus);
    const auto memory = parse_kib(read_text("/proc/meminfo"));
    for (const auto &name : {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}) {
      auto found = memory.find(name);
      if (found != memory.end()) values[std::string(name) + "_bytes"] = found->second;
    }
    if (memory.count("MemTotal") && memory.count("MemAvailable"))
      values["memory_used_bytes"] = memory.at("MemTotal") - memory.at("MemAvailable");
    if (memory.count("SwapTotal") && memory.count("SwapFree"))
      values["swap_used_bytes"] = memory.at("SwapTotal") - memory.at("SwapFree");
    const auto process = parse_kib(read_text("/proc/self/status"));
    if (process.count("VmRSS")) values["process_rss_bytes"] = process.at("VmRSS");
    const auto now = std::chrono::steady_clock::now();
    timespec usage{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &usage) == 0) {
      const double seconds = usage.tv_sec + usage.tv_nsec * 1e-9;
      if (last_ != std::chrono::steady_clock::time_point{})
        values["process_cpu_percent"] = 100 * (seconds - process_seconds_) /
            std::chrono::duration<double>(now - last_).count();
      process_seconds_ = seconds;
    }
    std::istringstream vm(read_text("/proc/vmstat"));
    std::string key;
    uint64_t pages;
    while (vm >> key >> pages) {
      if (key != "pswpin" && key != "pswpout") continue;
      if (swap_.count(key) && pages >= swap_[key] && last_ != std::chrono::steady_clock::time_point{})
        values[key + "_bytes_per_second"] = (pages - swap_[key]) * double(sysconf(_SC_PAGESIZE)) /
            std::chrono::duration<double>(now - last_).count();
      swap_[key] = pages;
    }
    last_ = now;
    return values;
  }
private:
  std::map<std::string, CpuTicks> previous_;
  std::map<std::string, uint64_t> swap_;
  std::chrono::steady_clock::time_point last_{};
  double process_seconds_ = 0;
};

inline std::vector<std::pair<size_t, size_t>> sampled_monitor_baselines(
    const std::vector<size_t> &receivers) {
  std::vector<std::pair<size_t, size_t>> pairs;
  for (size_t second = 1; second < receivers.size(); ++second)
    for (size_t first = 0; first < second; ++first)
      pairs.emplace_back(receivers[first], receivers[second]);
  if (pairs.size() <= 8) return pairs;
  std::vector<std::pair<size_t, size_t>> sample;
  for (size_t i = 0; i < 8; ++i)
    sample.push_back(pairs[i * (pairs.size() - 1) / 7]);
  return sample;
}

inline std::vector<std::pair<size_t, size_t>> monitor_baseline_pairs(
    size_t receivers, const std::string &selection) {
  std::vector<std::pair<size_t, size_t>> result;
  std::istringstream pairs(selection);
  std::string pair;
  while (std::getline(pairs, pair, ',')) {
    if (result.size() >= 16) throw std::invalid_argument("at most 16 monitor baselines");
    std::istringstream input(pair);
    int first, second;
    char separator, extra;
    if (!(input >> first >> separator >> second) || separator != ':' || (input >> extra) ||
        first < 0 || second < first || static_cast<size_t>(second) >= receivers)
      throw std::invalid_argument("monitor baselines require receiver indices 0 <= first <= second < receiver count");
    result.emplace_back(first, second);
  }
  if (result.empty() || selection.back() == ',')
    throw std::invalid_argument("monitor baselines require a nonempty receiver-pair list");
  return result;
}

struct Spectrum {
  size_t baseline;
  int pol;
  std::string name;
  bool valid;
  bool autocorrelation;
};
struct Snapshot {
  uint64_t first = 0, last = 0;
  int missing = 0, total = 0;
  int64_t sampled_ms = 0;
  std::vector<std::complex<float>> values;
};

// No CUDA or HDF5 dependency: also exercised by CPU-only tests.
class StarspinMonitor {
public:
  StarspinMonitor(std::string host, int port, double interval,
                  std::vector<double> frequencies, std::vector<Spectrum> spectra,
                  Json metadata)
      : publisher_(std::move(host), port), interval_(interval),
        frequencies_(std::move(frequencies)), spectra_(std::move(spectra)),
        metadata_(std::move(metadata)), run_(std::to_string(wall_ms()) + "-" + std::to_string(getpid())),
        prefix_("starspin:" + run_) {
    if (!std::isfinite(interval_) || interval_ < 1 || frequencies_.empty())
      throw std::invalid_argument("monitor interval must be finite and at least one second");
    metadata_["run_id"] = run_;
    metadata_["host_memory_scope"] = "/proc/meminfo (host view unless virtualised)";
    metadata_["cpu_scope"] = "/proc/stat logical CPUs; process CPU 100% = one core";
    metadata_["cgroup_v2_memory_max"] = read_text("/sys/fs/cgroup/memory.max");
    metadata_["cgroup_v2_cpu_max"] = read_text("/sys/fs/cgroup/cpu.max");
    metadata_["time_axis"] = "wall-clock sampling time; acquisition uses integration sample counters";
    publisher_.start();
    worker_ = std::thread([this] { run(); });
    std::fprintf(stderr, "Starspin monitoring run: %s\n", run_.c_str());
  }
  ~StarspinMonitor() {
    stopped_ = true;
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
    publisher_.stop();
  }
  void counters(Json values) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now = std::chrono::steady_clock::now();
    if (last_counters_ != std::chrono::steady_clock::time_point{}) {
      const double elapsed = std::chrono::duration<double>(now - last_counters_).count();
      for (const auto &name : {"packets_received", "packets_processed", "packets_missing", "packets_discarded", "nic_drops"})
        if (values.contains(name) && counters_.contains(name) && values[name] >= counters_[name])
          values[std::string(name) + "_per_second"] =
              (values[name].get<double>() - counters_[name].get<double>()) / elapsed;
    }
    last_counters_ = now;
    counters_ = std::move(values);
  }
  void written(size_t count) { written_ = count; last_write_ = wall_ms(); }
  template <typename Block> void observe(const Block &block) noexcept {
    ++completed_;
    const auto now = std::chrono::steady_clock::now();
    if (now < next_sample_) return;
    next_sample_ = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(interval_));
    try {
      Snapshot snapshot;
      snapshot.first = block.start_seq_num;
      snapshot.last = block.end_seq_num;
      snapshot.missing = block.num_missing_packets;
      snapshot.total = block.num_total_packets;
      snapshot.sampled_ms = wall_ms();
      snapshot.values.reserve(spectra_.size() * frequencies_.size());
      for (const auto &spectrum : spectra_)
        for (size_t channel = 0; channel < frequencies_.size(); ++channel) {
          const auto &v = block.data[channel][spectrum.baseline][spectrum.pol][spectrum.pol];
          snapshot.values.emplace_back(v[0], v[1]);
        }
      std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
      if (!lock) { ++skipped_; return; }
      if (pending_.sampled_ms) ++skipped_;
      pending_ = std::move(snapshot);
    } catch (const std::exception &) { ++skipped_; }
  }
  static Json summarize(const Snapshot &snapshot, size_t offset,
                        const std::vector<double> &frequencies, const Spectrum &spectrum) {
    Json rows = Json::array();
    const size_t stride = (frequencies.size() + 255) / 256;
    for (size_t begin = 0; begin < frequencies.size(); begin += stride) {
      const size_t end = std::min(begin + stride, frequencies.size());
      std::complex<double> sum{};
      double frequency = 0;
      bool valid = spectrum.valid && snapshot.total > 0 && snapshot.missing < snapshot.total;
      for (size_t channel = begin; channel < end; ++channel) {
        const auto value = snapshot.values.at(offset + channel);
        valid = valid && std::isfinite(value.real()) && std::isfinite(value.imag());
        sum += std::complex<double>(value);
        frequency += frequencies[channel];
      }
      sum /= double(end - begin);
      const double amplitude = std::abs(sum);
      rows.push_back({{"frequency_mhz", frequency / (end - begin) / 1e6},
                      // Redis datasource 2.2.0 mishandles JSON nulls. Numeric strings
                      // become numbers/NaN through Grafana's convertFieldType transform.
                      {"valid", valid},
                      {"power", valid && spectrum.autocorrelation ? Json(sum.real()).dump() : "NaN"},
                      {"amplitude", valid ? Json(amplitude).dump() : "NaN"},
                      {"phase_deg", valid && amplitude > 0 ? Json(std::arg(sum) * 180 / std::acos(-1)).dump() : "NaN"}});
    }
    return rows;
  }
private:
  void scalar(Batch &batch, const std::string &name, const Json &value, int64_t timestamp) {
    if (!value.is_number() || !std::isfinite(value.get<double>())) return;
    const auto key = prefix_ + ":" + name;
    batch.push_back({"TS.ADD", key, std::to_string(timestamp), value.dump(),
                     "RETENTION", "86400000", "ON_DUPLICATE", "LAST",
                     "LABELS", "run", run_, "metric", name, "category",
                     name.size() > 3 && name.compare(0, 3, "cpu") == 0 && std::isdigit(name[3]) &&
                     name.find("_busy_percent") != std::string::npos ? "cpu_core_busy" : "other"});
    batch.push_back({"EXPIRE", key, "86400"});
  }
  void json_value(Batch &batch, const std::string &key, const Json &value) {
    batch.push_back({"JSON.SET", key, ".", value.dump()});
    batch.push_back({"EXPIRE", key, "86400"});
  }
  void run() noexcept {
    HostSampler host;
    Json last_integration = Json::object();
    Json history = Json::array();
    while (!stopped_) {
      try {
        const int64_t now = wall_ms();
        Json values = host.sample();
        Snapshot snapshot;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          values.update(counters_);
          std::swap(snapshot, pending_);
        }
        values["integrations_completed"] = completed_.load();
        values["integrations_written"] = written_.load();
        if (last_write_) values["output_age_seconds"] = (now - last_write_.load()) / 1000.0;
        values["monitor_skipped"] = skipped_.load() + publisher_.dropped.load();
        values["redis_failures"] = publisher_.failures.load();
        values["heartbeat_ms"] = now;
        Batch batch;
        if (snapshot.sampled_ms) {
          last_integration = {{"first_sample", snapshot.first}, {"last_packet_sample", snapshot.last},
                              {"sampled_ms", snapshot.sampled_ms}, {"missing_packets", snapshot.missing},
                              {"total_packets", snapshot.total}};
          if (snapshot.total > 0)
            values["integration_missing_percent"] = 100.0 * snapshot.missing / snapshot.total;
          for (size_t i = 0; i < spectra_.size(); ++i) {
            const auto &spectrum = spectra_[i];
            auto rows = summarize(snapshot, i * frequencies_.size(), frequencies_, spectrum);
            json_value(batch, prefix_ + ":spectrum:" + spectrum.name, rows);
            if (spectrum.autocorrelation) {
              const bool valid = std::all_of(rows.begin(), rows.end(), [](const Json &bin) {
                return bin["valid"].get<bool>();
              });
              if (spectrum.name == metadata_.value("waterfall_input", std::string("rx0_p0"))) {
                for (const auto &bin : rows)
                  history.push_back({{"time", snapshot.sampled_ms}, {"frequency_mhz", bin["frequency_mhz"]}, {"power", bin["power"]}});
                const size_t limit = 120 * rows.size();
                if (history.size() > limit)
                  history.erase(history.begin(), history.begin() + (history.size() - limit));
                json_value(batch, prefix_ + ":waterfall", history);
              }
              if (valid) {
                double total = 0;
                for (size_t channel = 0; channel < frequencies_.size(); ++channel)
                  total += snapshot.values[i * frequencies_.size() + channel].real();
                scalar(batch, "power_" + spectrum.name, total / frequencies_.size(), snapshot.sampled_ms);
              }
            }
          }
        }
        if (last_integration.contains("sampled_ms"))
          values["integration_age_seconds"] = (now - last_integration["sampled_ms"].get<int64_t>()) / 1000.0;
        for (const auto &[name, value] : values.items()) scalar(batch, name, value, now);
        Json status = metadata_;
        status["last_integration"] = last_integration;
        status["published_sample_ms"] = now;
        status["update_interval_seconds"] = interval_;
        Json flat_status = Json::array();
        for (const auto &[key, value] : status.items())
          flat_status.push_back({{"field", key}, {"value", value.is_string() ? value.get<std::string>() : value.dump()}});
        json_value(batch, prefix_ + ":status", flat_status);
        const auto observation = metadata_.value("observation", Json::object());
        json_value(batch, prefix_ + ":calibration", Json::array({{
            {"boot", observation.value("boot_verification", Json("unverified")).dump()},
            {"delay_closure", observation.value("delay_closure", Json("unverified")).dump()}}}));
        Command inputs{"SADD", prefix_ + ":inputs"}, baselines{"SADD", prefix_ + ":baselines"};
        for (const auto &spectrum : spectra_)
          (spectrum.autocorrelation ? inputs : baselines).push_back(spectrum.name);
        if (inputs.size() > 2) batch.push_back(std::move(inputs));
        if (baselines.size() > 2) batch.push_back(std::move(baselines));
        batch.push_back({"EXPIRE", prefix_ + ":inputs", "86400"});
        batch.push_back({"EXPIRE", prefix_ + ":baselines", "86400"});
        batch.push_back({"SET", "starspin:latest", run_, "EX", "86400"});
        publisher_.submit(std::move(batch));
      } catch (const std::exception &e) {
        ++skipped_;
        std::fprintf(stderr, "Monitoring sample skipped: %s\n", e.what());
      }
      std::unique_lock<std::mutex> lock(wait_mutex_);
      wake_.wait_for(lock, std::chrono::duration<double>(interval_), [this] { return stopped_.load(); });
    }
  }
  RedisPublisher publisher_;
  double interval_;
  std::vector<double> frequencies_;
  std::vector<Spectrum> spectra_;
  Json metadata_, counters_ = Json::object();
  std::string run_, prefix_;
  std::mutex mutex_, wait_mutex_;
  std::condition_variable wake_;
  Snapshot pending_;
  std::atomic<uint64_t> completed_{0}, written_{0}, skipped_{0};
  std::atomic<int64_t> last_write_{0};
  std::atomic<bool> stopped_{false};
  std::chrono::steady_clock::time_point next_sample_{}, last_counters_{};
  std::thread worker_;
};
} // namespace spatial_monitor
