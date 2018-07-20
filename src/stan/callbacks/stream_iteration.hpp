#ifndef STAN_CALLBACKS_STREAM_ITERATION_HPP
#define STAN_CALLBACKS_STREAM_ITERATION_HPP

#include <stan/callbacks/iteration.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <cmath>

namespace stan {
namespace callbacks {

/**
 * <code>stream_iteration</code> is an implementation
 * of <code>iteration</code> that writes to a stream.
 */
class stream_iteration : public iteration {
 public:

  /**
   * Constructs a <code>stream_iteration</code> with an output stream
   * and an optional prefix for comments.
   *
   * @param[in, out] output stream to write
   * @param[in] comment_prefix string to stream before
   *   each comment line. Default is "".
   */
  stream_iteration(std::ostream &output,
                   int num_warmup_iterations,
                   int num_total_iterations,
                   int refresh_iterations = 0)
      : logger_(stream_logger(output, output, output, output, output)),
        num_warmup_iterations_(num_warmup_iterations),
        num_total_iterations_(num_total_iterations),
        refresh_iterations_(refresh_iterations) {
  }

  explicit stream_iteration(stream_logger &logger,
                            int num_warmup_iterations,
                            int num_total_iterations,
                            int refresh_iterations = 0)
      : logger_(logger),
        num_warmup_iterations_(num_warmup_iterations),
        num_total_iterations_(num_total_iterations),
        refresh_iterations_(refresh_iterations) {
  }

  virtual bool should_print(int iteration_number) {
    if (refresh_iterations_ == 0)
      return false;
    if (iteration_number == 1
        || iteration_number == num_total_iterations_
        || iteration_number == num_warmup_iterations_
        || iteration_number % refresh_iterations_ == 0)
      return true;
    return false;
  }

  /**
   *
   */
  virtual void operator()(int iteration_number) {
    if (should_print(iteration_number)) {
      int print_width
          = std::ceil(std::log10(static_cast<double>(num_total_iterations_)));
      std::stringstream message;
      message << "Iteration: ";
      message << std::setw(print_width) << iteration_number << " / " << num_total_iterations_;
      message << " [" << std::setw(3)
              << static_cast<int>((100.0 * iteration_number) / num_total_iterations_)
              << "%] ";
      message << (iteration_number <= num_warmup_iterations_
                  ? " (Warmup)" : " (Sampling)");
      logger_.info(message);
    }
  }

  /**
   * Virtual destructor
   */
  virtual ~stream_iteration() {}

 private:
  /**
   * Output stream
   */
  stream_logger logger_;
  int num_warmup_iterations_;
  int num_total_iterations_;
  const int refresh_iterations_;
};
}
}

#endif
