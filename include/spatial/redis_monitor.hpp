#pragma once

#include <hiredis/hiredis.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace spatial_monitor {
using Command = std::vector<std::string>;
using Batch = std::vector<Command>;

// Only this worker touches Redis. Producers replace a single pending batch;
// a failed server can never fill a scientific-output ring with network work.
class RedisPublisher {
public:
  explicit RedisPublisher(std::string host = "127.0.0.1", int port = 6379)
      : host_(std::move(host)), port_(port) {
    in_addr v4{};
    in6_addr v6{};
    if ((inet_pton(AF_INET, host_.c_str(), &v4) != 1 &&
         inet_pton(AF_INET6, host_.c_str(), &v6) != 1) || port < 1 || port > 65535)
      throw std::invalid_argument("monitor Redis requires a numeric IP and port 1..65535");
  }
  ~RedisPublisher() { stop(); }
  RedisPublisher(const RedisPublisher &) = delete;
  RedisPublisher &operator=(const RedisPublisher &) = delete;

  // Register once, before start(). Replayed after reconnect; no network I/O here.
  using Create = std::function<void(const Command &)>;
  void setup(std::function<void(const Create &)> generate) { setup_ = std::move(generate); }
  void start() noexcept {
    try { worker_ = std::thread([this] { run(); }); }
    catch (const std::exception &e) {
      stopped_ = true;
      std::fprintf(stderr, "Redis monitoring disabled: %s\n", e.what());
    }
  }
  void stop() noexcept {
    stopped_ = true;
    wake_.notify_all();
    {
      std::lock_guard<std::mutex> lock(socket_mutex_);
      if (socket_ >= 0) shutdown(socket_, SHUT_RDWR);
    }
    if (worker_.joinable()) worker_.join();
  }
  void submit(Batch batch) noexcept {
    size_t bytes = 0;
    for (const auto &command : batch)
      for (const auto &arg : command) bytes += arg.size();
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock || stopped_ || bytes > 16 * 1024 * 1024) { ++dropped; return; }
    if (!pending_.empty()) ++dropped;
    pending_.swap(batch);
    wake_.notify_one();
  }
  std::atomic<uint64_t> dropped{0}, failures{0}, published{0};
  std::atomic<bool> connected{false};

private:
  using Context = std::unique_ptr<redisContext, decltype(&redisFree)>;
  struct Stopped {};
  void execute(redisContext *context, const Command &command, bool setup) {
    if (stopped_) throw Stopped{};
    std::vector<const char *> argv;
    std::vector<size_t> lengths;
    for (const auto &arg : command) {
      argv.push_back(arg.data());
      lengths.push_back(arg.size());
    }
    std::unique_ptr<redisReply, decltype(&freeReplyObject)> reply(
        static_cast<redisReply *>(redisCommandArgv(
            context, static_cast<int>(argv.size()), argv.data(), lengths.data())),
        freeReplyObject);
    if (!reply) throw std::runtime_error(context->errstr);
    check_reply(reply.get(), setup);
  }
  void check_reply(const redisReply *reply, bool setup) {
    if (reply->type == REDIS_REPLY_ARRAY)
      for (size_t i = 0; i < reply->elements; ++i) check_reply(reply->element[i], setup);
    if (reply->type == REDIS_REPLY_ERROR) {
      const std::string error(reply->str, reply->len);
      // Setup is idempotent across reconnects. All other errors trigger retry.
      if (setup && (error.find("already exists") != std::string::npos ||
                    error.find("already has a compaction rule") != std::string::npos ||
                    error.find("destination key already has a src rule") != std::string::npos)) return;
      throw std::runtime_error(error);
    }
  }
  void run() noexcept {
    Context context(nullptr, redisFree);
    auto last_warning = std::chrono::steady_clock::time_point{};
    while (!stopped_) {
      Batch batch;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [this] { return stopped_ || !pending_.empty(); });
        if (stopped_) break;
        batch.swap(pending_);
      }
      try {
        if (!context) {
          const timeval timeout{0, 100000};
          context.reset(redisConnectWithTimeout(host_.c_str(), port_, timeout));
          if (!context || context->err)
            throw std::runtime_error(context ? context->errstr : "connection allocation failed");
          {
            std::lock_guard<std::mutex> lock(socket_mutex_);
            // Own a duplicate: hiredis may close its fd after an I/O error.
            socket_ = dup(context->fd);
            if (socket_ < 0) throw std::runtime_error("could not duplicate Redis socket");
          }
          if (redisSetTimeout(context.get(), timeout) != REDIS_OK)
            throw std::runtime_error("could not set Redis socket timeout");
          if (setup_) setup_([&](const Command &command) {
            execute(context.get(), command, true);
          });
          connected = !stopped_;
        }
        // Publish spectra and their sample identity together. Disconnecting
        // before EXEC discards the transaction, including during shutdown.
        if (batch.size() > 1) execute(context.get(), {"MULTI"}, false);
        for (const auto &command : batch) {
          if (stopped_) break;
          execute(context.get(), command, false);
        }
        if (batch.size() > 1 && !stopped_) execute(context.get(), {"EXEC"}, false);
        if (!stopped_) ++published;
      } catch (const Stopped &) {
        break;
      } catch (const std::exception &e) {
        if (stopped_) break;
        ++failures;
        ++dropped;
        connected = false;
        disconnect(context);
        const auto now = std::chrono::steady_clock::now();
        if (now - last_warning >= std::chrono::seconds(30)) {
          std::fprintf(stderr, "Redis monitoring unavailable (capture continues): %s\n", e.what());
          last_warning = now;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait_for(lock, std::chrono::seconds(1), [this] { return stopped_.load(); });
      }
    }
    disconnect(context);
    connected = false;
  }
  void disconnect(Context &context) {
    std::lock_guard<std::mutex> lock(socket_mutex_);
    if (socket_ >= 0) close(socket_);
    socket_ = -1;
    context.reset();
  }
  std::string host_;
  int port_;
  std::function<void(const Create &)> setup_;
  Batch pending_;
  std::atomic<bool> stopped_{false};
  std::mutex mutex_, socket_mutex_;
  int socket_ = -1;
  std::condition_variable wake_;
  std::thread worker_;
};
} // namespace spatial_monitor
