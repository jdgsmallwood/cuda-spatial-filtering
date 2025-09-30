#pragma once
#include "spdlog/sinks/stdout_color_sinks.h"
#include <memory>
#include <spdlog/spdlog.h>
#define LOG_INFO(...) spatial::Logger::get()->info(__VA_ARGS__)
#define LOG_DEBUG(...) spatial::Logger::get()->debug(__VA_ARGS__)
#define LOG_WARN(...) spatial::Logger::get()->warn(__VA_ARGS__)
#define LOG_ERROR(...) spatial::Logger::get()->error(__VA_ARGS__)

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
