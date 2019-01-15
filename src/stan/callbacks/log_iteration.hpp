#ifndef STAN_CALLBACKS_LOG_ITERATION_HPP
#define STAN_CALLBACKS_LOG_ITERATION_HPP

#include <stan/callbacks/iteration.hpp>
#include <stan/callbacks/logger.hpp>
#include <cmath>

namespace stan {
namespace callbacks {

/**
 * <code>log_iteration</code> is an implementation
 * of <code>iteration</code> that writes to a logger.
 */
class log_iteration : public iteration {
 public:
  /**
   * Constructs a <code>log_iteration</code> with a logger.
   * 
   * This will write iteration messages to the logger at the info level.
   *
   * @param[in, out] logger a logger to log output to
   * @param[in] num_warmup_iterations number of warmup iterations
   * @param[in] num_total_iterations number of total iterations (including)
   *   warmup
   * @param[in] refresh_iterations number of iterations before printing again.
   *   This number must be greater or equal to 0. When this is set to 0, no
   *   messages are logged.
   */
  explicit log_iteration(logger &logger,
                         int num_warmup_iterations,
                         int num_total_iterations,
                         int refresh_iterations = 0)
      : logger_(logger),
        num_warmup_iterations_(num_warmup_iterations),
        num_total_iterations_(num_total_iterations),
        refresh_iterations_(refresh_iterations) {
  }

  /**
   * Returns true if this iteration should print a 
   * message to the logger.
   * 
   * This virtual method indicates whether the
   * iteration should print. This allows us to
   * build subclasses that can change behavior
   * by just implementing this method.
   * 
   * The default implementation prints at the first iteration,
   * the last warmup iteration, the last iteration, or if
   * the iteration number modulus refresh_iterations is 0.
   *
   * If refresh_iterations is 0, this will not print.
   * 
   * @param iteration_number current iteration
   * @return true if it should print, false otherwise
   */
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
   * Writes iteration message to the logger as info.
   * 
   * If the <code>should_print</code> method returns true,
   * this prints a message to the logger as info.
   *
   * The message is formatted as:
   * "Iteration: <iteration_number> / <num_total_iterations> [<percentage>%] ({Warmup,Sampling})"
   * 
   * @param iteration_number current iteration number (before the iteration starts)
   */
  virtual void operator()(int iteration_number) {
    if (should_print(iteration_number)) {
      int print_width
          = std::ceil(std::log10(static_cast<double>(num_total_iterations_)));
      std::stringstream message;
      message << "Iteration: ";
      message << std::setw(print_width) << iteration_number << " / "
              << num_total_iterations_;
      message << " [" << std::setw(3)
              << static_cast<int>((100.0 * iteration_number)
                                  / num_total_iterations_)
              << "%] ";
      message << (iteration_number <= num_warmup_iterations_
                  ? " (Warmup)" : " (Sampling)");
      logger_.info(message);
    }
  }

  /**
   * Virtual destructor
   */
  virtual ~log_iteration() {}

 private:
  logger& logger_;
  int num_warmup_iterations_;
  int num_total_iterations_;
  const int refresh_iterations_;
};
}
}

#endif
