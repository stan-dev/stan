#ifndef STAN_CALLBACKS_STREAM_TIMED_ITERATIONS_HPP
#define STAN_CALLBACKS_STREAM_TIMED_ITERATIONS_HPP

#include <stan/callbacks/stream_iteration.hpp>
#include <chrono>
#include <cmath>

namespace stan {
namespace callbacks {

/**
 * <code>stream_iteration</code> is an implementation
 * of <code>iteration</code> that writes to a stream.
 */
class stream_timed_iteration : public stream_iteration {
 public:

  /**
   * Constructs a <code>stream_iteration</code> with an output stream
   * and an optional prefix for comments.
   *
   * @param[in, out] output stream to write
   * @param[in] comment_prefix string to stream before
   *   each comment line. Default is "".
   */
  stream_timed_iteration(std::ostream &output,
                         int num_warmup_iterations,
                         int num_total_iterations,
                         double refresh_seconds)
      : stream_iteration(output, num_warmup_iterations, num_total_iterations),
        refresh_seconds_(refresh_seconds),
        last_time_() {
  }

  stream_timed_iteration(stream_logger &logger,
                         int num_warmup_iterations,
                         int num_total_iterations,
                         double refresh_seconds)
      : stream_iteration(logger, num_warmup_iterations, num_total_iterations),
        refresh_seconds_(refresh_seconds),
        last_time_() {
  }

  bool should_print(int iteration_number) {
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
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
