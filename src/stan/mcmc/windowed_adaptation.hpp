#ifndef STAN_MCMC_WINDOWED_ADAPTATION_HPP
#define STAN_MCMC_WINDOWED_ADAPTATION_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/base_adaptation.hpp>
#include <ostream>
#include <string>

namespace stan {
namespace mcmc {
/* Warmup schedule for NUTS-HMC windowed adaptation has 3 phases.
 * Phases 1 & 3: have a fixed size.
 * Phase 2 iterations are divided into "windows" which double in length so that
 * phase 2 iterations fill out the total number of warmup iterations.
 */
class windowed_adaptation : public base_adaptation {
 public:
  explicit windowed_adaptation(std::string name) : estimator_name_(name) {
    num_warmup_ = 0;
    cur_iter_ = 0;
    cur_phase2_ = 1;
    cur_phase2_end_ = 0;
    end_phase1_ = 0;
    start_phase3_ = 0;
  }

  /* Record user requested number of warmup iterations and adjust size of
   * warmup phases as needed.
   *
   * @param[in] num_warmup number of warmup draws
   * @param[in] init_buffer width of initial fast adaptation interval
   * @param[in] term_buffer width of final fast adaptation interval
   * @param[in] base_window initial width of slow adaptation interval
   * @param[in,out] logger Logger for messages
   */
  void set_window_params(unsigned int num_warmup, unsigned int init_buffer,
                         unsigned int term_buffer, unsigned int base_window,
                         callbacks::logger& logger) {
    num_warmup_ = num_warmup - 1;  // count from 0
    end_phase1_ = (init_buffer > 0) ? 0 : init_buffer - 1;
    start_phase3_ = num_warmup - term_buffer;
    cur_phase2_ = base_window;
    cur_phase2_end_ = end_phase1_ + cur_phase2_;

    if (init_buffer + base_window + term_buffer > num_warmup) {
      logger.info(
          "WARNING: There aren't enough warmup "
          "iterations to fit the");
      logger.info(
          "         three stages of adaptation as currently"
          " configured.");

      num_warmup_ = num_warmup;
      end_phase1_ = 0.15 * num_warmup;  // C++ rounds down
      start_phase3_ = num_warmup - (0.10 * num_warmup);
      cur_phase2_ = (base_window <= 0.75 * num_warmup)
                        ? base_window
                        : start_phase3_ - end_phase1_;
      cur_phase2_end_ = end_phase1_ + cur_phase2_;

      logger.info(
          "         Reducing each adaptation stage to "
          "15%/75%/10% of");
      logger.info("         the given number of warmup iterations:");

      std::stringstream init_buffer_msg;
      init_buffer_msg << "           init_buffer = " << end_phase1_;
      logger.info(init_buffer_msg);

      std::stringstream adapt_window_msg;
      adapt_window_msg << "           adapt_window = "
                       << start_phase3_ - end_phase1_;
      logger.info(adapt_window_msg);

      std::stringstream term_buffer_msg;
      term_buffer_msg << "           term_buffer = "
                      << num_warmup - start_phase3_;
      logger.info(term_buffer_msg);

      logger.info("");
    }
  }

  bool in_phase2_window() {
    return (cur_iter_ > end_phase1_ && cur_iter_ < start_phase3_);
  }

  bool end_phase2_window() { return (cur_iter_ == cur_phase2_end_); }

  // find next window endpoint
  // double window size if possible, else use remaining phase2 iters
  void compute_next_window() {
    int next_phase2_size = cur_phase2_ * 2;
    if (next_phase2_size + cur_iter_ <= start_phase3_) {
      cur_phase2_ = next_phase2_size;
      cur_phase2_end_ = cur_iter_ + cur_phase2_;
    } else {
      cur_phase2_end_ = start_phase3_;
    }
  }

  unsigned int cur_iter_;

 protected:
  std::string estimator_name_;

  unsigned int num_warmup_;
  unsigned int cur_phase2_;
  unsigned int cur_phase2_end_;
  unsigned int end_phase1_;
  unsigned int start_phase3_;
};

}  // namespace mcmc
}  // namespace stan
#endif
