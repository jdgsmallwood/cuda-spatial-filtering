#include "spatial/starspin_monitor.hpp"
#include <cassert>
#include <iostream>
#include <thread>

using namespace spatial_monitor;
using Clock = std::chrono::steady_clock;
struct Block {
  float data[2][3][2][2][2]{};
  uint64_t start_seq_num = 100, end_seq_num = 164;
  int num_missing_packets = 0, num_total_packets = 4;
};

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "publish") {
    const int port = std::stoi(argv[2]);
    const double seconds = std::stod(argv[3]);
    const auto start = Clock::now();
    {
      StarspinMonitor monitor("127.0.0.1", port, 1, {100e6, 101e6},
          {{0, 0, "rx0_p0", true, true}, {1, 0, "rx0_rx1_p0", true, false}},
          {{"waterfall_input", "rx0_p0"}, {"observation", {{"boot_verification", "unverified"}}}});
      Block block;
      block.data[0][0][0][0][0] = 4;
      block.data[1][0][0][0][0] = 8;
      block.data[0][1][0][0][0] = 3;
      block.data[0][1][0][0][1] = 4;
      while (std::chrono::duration<double>(Clock::now() - start).count() < seconds) {
        block.data[0][0][0][0][0] = 4;
        monitor.observe(block);
        block.data[0][0][0][0][0] = 999; // worker must own its copy
        monitor.counters({{"packets_received", 100}, {"packets_processed", 100},
                          {"packets_missing", 0}, {"packets_discarded", 0},
                          {"nic_drops", 0}, {"pipeline_backlog", 0}});
        monitor.written(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    assert(std::chrono::duration<double>(Clock::now() - start).count() < seconds + 2);
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "publisher") {
    const auto start = Clock::now();
    {
      RedisPublisher publisher("127.0.0.1", std::stoi(argv[2]));
      publisher.setup([](const auto &create) { create({"SET", "monitor-test:setup", "1"}); });
      publisher.start();
      for (int i = 0; i < 300; ++i) {
        const auto before = Clock::now();
        publisher.submit({{"SET", "monitor-test:latest", std::to_string(i)},
                          {"SET", "monitor-test:heartbeat", std::to_string(i)}});
        assert(Clock::now() - before < std::chrono::milliseconds(100));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      std::cout << "published=" << publisher.published << " failures=" << publisher.failures
                << " dropped=" << publisher.dropped << '\n';
    }
    assert(Clock::now() - start < std::chrono::seconds(5));
    return 0;
  }
  const auto cpus = parse_cpu("cpu 10 2 3 80 5 0 0 0 999 999\ncpu0 1 0 2 7 0 0 0 0\nctxt 123\n");
  assert(cpus.size() == 2);
  const auto delta = cpu_delta({}, cpus.at("cpu"));
  assert(delta.at("user") == 12);
  assert(delta.at("system") == 3);
  assert(delta.at("busy") == 15);
  assert(cpu_delta(cpus.at("cpu"), {}).empty());
  assert(cpu_delta({}, {}).empty());
  assert(parse_kib("MemAvailable: 123 kB\nNoUnits: 10\n").at("MemAvailable") == 123 * 1024);
  bool bad_host = false;
  try { RedisPublisher bad("hostname.invalid"); } catch (const std::invalid_argument &) { bad_host = true; }
  assert(bad_host);
  Snapshot snapshot;
  snapshot.total = 10;
  snapshot.values = {{3,4}, {0,0}};
  const Spectrum cross{1, 0, "cross", true, false};
  auto result = StarspinMonitor::summarize(snapshot, 0, {100e6, 101e6}, cross);
  assert(result[0]["frequency_mhz"] == 100);
  assert(std::stod(result[0]["amplitude"].get<std::string>()) == 5);
  assert(std::abs(std::stod(result[0]["phase_deg"].get<std::string>()) - 53.130102) < 1e-5);
  assert(result[1]["phase_deg"] == "NaN");
  assert(result[0]["power"] == "NaN");
  snapshot.missing = snapshot.total;
  result = StarspinMonitor::summarize(snapshot, 0, {100e6, 101e6}, cross);
  assert(!result[0]["valid"].get<bool>());
  assert(result[0]["amplitude"] == "NaN");
  snapshot.missing = 0;
  snapshot.values.assign(513, {4,0});
  result = StarspinMonitor::summarize(snapshot, 0, std::vector<double>(513, 100e6), {0,0,"auto",true,true});
  assert(result.size() <= 256);
  assert(std::stod(result.back()["power"].get<std::string>()) == 4);
  {
    RedisPublisher publisher;
    publisher.submit({{"SET", "large", std::string(16 * 1024 * 1024, 'x')}});
    assert(publisher.dropped == 1);
    publisher.submit({{"SET", "key", "old"}});
    publisher.submit({{"SET", "key", "new"}});
    assert(publisher.dropped == 2); // at most one pending update, even before start
  }
  HostSampler sampler;
  auto first = sampler.sample();
  assert(!first.contains("cpu_busy_percent"));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto second = sampler.sample();
  assert(second.contains("cpu_busy_percent"));
  assert(second.contains("process_rss_bytes"));
  assert(second.at("process_rss_bytes").get<uint64_t>() > 0);
  assert(second.contains("process_cpu_percent"));
  std::cout << "monitor unit checks passed\n";
}
