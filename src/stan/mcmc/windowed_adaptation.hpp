#ifndef STAN_MCMC_WINDOWED_ADAPTATION_HPP
#define STAN_MCMC_WINDOWED_ADAPTATION_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/base_adaptation.hpp>
#include <ostream>
#include <string>

namespace stan {
namespace mcmc {

class windowed_adaptation : public base_adaptation {
 public:
  template <typename T>
  using require_string_convertible = std::enable_if_t<
      std::is_convertible<std::decay_t<T>, std::string>::value>;

  template <typename T, require_string_convertible<T>...>
  explicit windowed_adaptation(T&& name)
      : estimator_name_(std::forward<T>(name)) {}

  inline void restart() {
    adapt_window_counter_ = 0;
    adapt_window_size_ = adapt_base_window_;
    if ((adapt_init_buffer_ + adapt_window_size_) == 0) {
      adapt_next_window_ = num_warmup_;
    } else {
      adapt_next_window_ = adapt_init_buffer_ + adapt_window_size_ - 1;
    }
  }

  /**
   * Set the number of window parameters
   */
  inline void set_window_params(unsigned int num_warmup,
                                unsigned int init_buffer,
                                unsigned int term_buffer,
                                unsigned int base_window,
                                callbacks::logger& logger) {
    if (num_warmup < 20) {
      logger.info("WARNING: No " + estimator_name_ + " estimation is");
      logger.info("         performed for num_warmup < 20");
      logger.info("");
      return;
    }

    if (init_buffer + base_window + term_buffer > num_warmup) {
      logger.info(
          "WARNING: There aren't enough warmup "
          "iterations to fit the");
      logger.info("         three stages of adaptation as currently"
                  + std::string(" configured."));

      num_warmup_ = num_warmup;
      adapt_init_buffer_ = 0.15 * num_warmup;
      adapt_term_buffer_ = 0.10 * num_warmup;
      adapt_base_window_
          = num_warmup - (adapt_init_buffer_ + adapt_term_buffer_);

      logger.info(
          "         Reducing each adaptation stage to "
          "15%/75%/10% of");
      logger.info("         the given number of warmup iterations:");

      std::stringstream init_buffer_msg;
      init_buffer_msg << "           init_buffer = " << adapt_init_buffer_;
      logger.info(init_buffer_msg);

      std::stringstream adapt_window_msg;
      adapt_window_msg << "           adapt_window = " << adapt_base_window_;
      logger.info(adapt_window_msg);

      std::stringstream term_buffer_msg;
      term_buffer_msg << "           term_buffer = " << adapt_term_buffer_;
      logger.info(term_buffer_msg);

      logger.info("");
      return;
    }

    num_warmup_ = num_warmup;
    adapt_init_buffer_ = init_buffer;
    adapt_term_buffer_ = term_buffer;
    adapt_base_window_ = base_window;
    restart();
  }

  inline bool adaptation_window() {
    return (adapt_window_counter_ >= adapt_init_buffer_)
           && (adapt_window_counter_ < num_warmup_ - adapt_term_buffer_)
           && (adapt_window_counter_ != num_warmup_);
  }

  inline bool end_adaptation_window() {
    return (adapt_window_counter_ == adapt_next_window_)
           && (adapt_window_counter_ != num_warmup_);
  }

  inline void compute_next_window() {
    if (adapt_next_window_ == num_warmup_ - adapt_term_buffer_ - 1)
      return;
    adapt_window_size_ *= 2;
    adapt_next_window_ = adapt_window_counter_ + adapt_window_size_;
    if (adapt_next_window_ == num_warmup_ - adapt_term_buffer_ - 1)
      return;

    // Boundary of the following window, not the window just computed
    unsigned int next_window_boundary
        = adapt_next_window_ + 2 * adapt_window_size_;

    // If the following window overtakes the full adaptation window,
    // then stretch the current window to the end of the full window
    if (next_window_boundary >= num_warmup_ - adapt_term_buffer_) {
      adapt_next_window_ = num_warmup_ - adapt_term_buffer_ - 1;
    }
  }

 protected:
  std::string estimator_name_;

  unsigned int num_warmup_{1};
  unsigned int adapt_init_buffer_{0};
  unsigned int adapt_term_buffer_{0};
  unsigned int adapt_base_window_{0};

  unsigned int adapt_window_counter_{0};
  unsigned int adapt_next_window_{0};
  unsigned int adapt_window_size_{0};
};

}  // namespace mcmc
}  // namespace stan
#endif
