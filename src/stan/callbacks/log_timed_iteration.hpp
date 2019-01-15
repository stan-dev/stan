#ifndef STAN_CALLBACKS_LOG_TIMED_ITERATIONS_HPP
#define STAN_CALLBACKS_LOG_TIMED_ITERATIONS_HPP

#include <stan/callbacks/log_iteration.hpp>
#include <chrono>  // NOLINT(build/c++11)
#include <cmath>

namespace stan {
namespace callbacks {

/**
 * <code>log_timed_iteration</code> is an implementation
 * of <code>log_iteration</code> that writes to a logger.
 */
class log_timed_iteration : public log_iteration {
 public:
  /**
   * Constructs a <code>log_timed_iteration</code> with a logger.
   * 
   * This will write iteration messages to the logger at the info level.
   *
   * @param[in, out] logger a logger to log output to
   * @param[in] num_warmup_iterations number of warmup iterations
   * @param[in] num_total_iterations number of total iterations (including)
   *   warmup
   * @param[in] refresh_seconds seconds between log messages
   */
  log_timed_iteration(logger &logger,
                      int num_warmup_iterations,
                      int num_total_iterations,
                      double refresh_seconds)
      : log_iteration(logger, num_warmup_iterations, num_total_iterations),
        refresh_seconds_(refresh_seconds),
        last_time_() {
  }

  /**
   * Indicates that it should print when the last message
   * is greater than refresh_seconds.
   * 
   * @param iteration_number the current iteration number
   */
  bool should_print(int iteration_number) {
    std::chrono::system_clock::time_point now
      = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = now - last_time_;
    if (elapsed_seconds.count() < refresh_seconds_)
      return false;
    last_time_ = std::chrono::system_clock::now();
    return true;
  }

 private:
  const double refresh_seconds_;
  std::chrono::system_clock::time_point last_time_;
};
}
}

#endif
