#include "spatial/ingest_trace.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <sstream>
#include <thread>

TEST(CompletionInterval, DefaultAndValidatedOverrides) {
  EXPECT_EQ(spatial::completion_poll_interval_us(nullptr), 200);
  EXPECT_EQ(spatial::completion_poll_interval_us("1000"), 1000);
  EXPECT_EQ(spatial::completion_poll_interval_us("50"), 50);
  EXPECT_EQ(spatial::completion_poll_interval_us("10000"), 10000);
  for (const char *bad : {"", "0", "49", "10001", "1000junk", "-1",
                          "999999999999999999999999"})
    EXPECT_THROW(spatial::completion_poll_interval_us(bad), std::invalid_argument);
}

TEST(CoordinatorAffinity, DefaultAndValidatedOverrides) {
  EXPECT_EQ(spatial::coordinator_cpu_from_env(nullptr), -1);
  EXPECT_EQ(spatial::coordinator_cpu_from_env("0"), 0);
  EXPECT_EQ(spatial::coordinator_cpu_from_env("7"), 7);
  for (const char *bad : {"", "-1", "7,8", "7junk", "99999999999999999999"})
    EXPECT_THROW(spatial::coordinator_cpu_from_env(bad), std::invalid_argument);
  EXPECT_THROW(spatial::coordinator_cpu_from_env(std::to_string(CPU_SETSIZE).c_str()),
               std::invalid_argument);
}

TEST(IngestTrace, BoundedIndependentLanesExportAfterJoin) {
  const auto path = std::filesystem::temp_directory_path() /
                   ("ingest_trace_test_" + std::to_string(getpid()) + ".csv");
  std::filesystem::remove(path);
  spatial::IngestTrace trace(path.c_str(), 3);
  std::thread a([&] {
    for (uint64_t i = 0; i < 5; ++i)
      trace.record(1, spatial::IngestTrace::WorkerPause, i * 100, i * 100 + 10);
  });
  std::thread b([&] {
    trace.record(5, spatial::IngestTrace::ReceiveRingWait, 205, 350, 1536);
  });
  a.join(); b.join();
  EXPECT_FALSE(std::filesystem::exists(path));
  EXPECT_EQ(trace.lane(1).written, 5);
  EXPECT_EQ(trace.lane(5).written, 1);
  EXPECT_NE(trace.lane(1).tid, trace.lane(5).tid);
  trace.dump();
  std::ifstream file(path);
  std::stringstream text; text << file.rdbuf();
  const auto csv = text.str();
  EXPECT_NE(csv.find("written=5 overwritten=2"), std::string::npos);
  EXPECT_NE(csv.find(",2,200,210,0,0,0\n"), std::string::npos);
  EXPECT_NE(csv.find(",2,400,410,0,0,0\n"), std::string::npos);
  EXPECT_EQ(csv.find(",2,0,10,0,0,0\n"), std::string::npos);
  EXPECT_NE(csv.find(",4,205,350,1536,0,0\n"), std::string::npos);
  std::filesystem::remove(path);
}

TEST(IngestTrace, DisabledDoesNotAllocateOrRecord) {
  spatial::IngestTrace trace(nullptr);
  trace.record(0, spatial::IngestTrace::CompletionCycle, 0, 1);
  EXPECT_FALSE(trace.enabled());
  EXPECT_TRUE(trace.lane(0).records.empty());
  EXPECT_EQ(trace.lane(0).written, 0);
  EXPECT_NO_THROW(trace.dump());
}
