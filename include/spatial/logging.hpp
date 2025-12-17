#pragma once
#include "spdlog/sinks/stdout_color_sinks.h"
#include <memory>
#include <spdlog/spdlog.h>
#define LOG_INFO(...) spatial::Logger::get()->info(__VA_ARGS__)
#define LOG_DEBUG(...) spatial::Logger::get()->debug(__VA_ARGS__)
#define LOG_WARN(...) spatial::Logger::get()->warn(__VA_ARGS__)
#define LOG_ERROR(...) spatial::Logger::get()->error(__VA_ARGS__)
#define FLUSH_LOG() spatial::Logger::get()->flush();

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", #call, __FILE__,      \
              __LINE__, cudaGetErrorString(err));                              \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUSOLVER_CHECK(err)                                                    \
  do {                                                                         \
    cusolverStatus_t err_ = (err);                                             \
    if (err_ != CUSOLVER_STATUS_SUCCESS) {                                     \
      printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);        \
    }                                                                          \
  } while (0)

namespace spatial {

class Logger {
public:
  static std::shared_ptr<spdlog::logger> &get() {
    static std::shared_ptr<spdlog::logger> logger =
        spdlog::stdout_color_mt("spatial");
    return logger;
  }

  // Allow the application to set a custom logger
  static void set(std::shared_ptr<spdlog::logger> custom_logger) {
    get() = std::move(custom_logger);
  }
};

} // namespace spatial
