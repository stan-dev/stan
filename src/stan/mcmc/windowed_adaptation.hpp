#ifndef STAN_MCMC_WINDOWED_ADAPTATION_HPP
#define STAN_MCMC_WINDOWED_ADAPTATION_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/base_adaptation.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace mcmc {

    class windowed_adaptation: public base_adaptation {
    public:
      explicit windowed_adaptation(std::string name)
        : estimator_name_(name) {
        num_warmup_ = 0;
        adapt_init_buffer_ = 0;
        adapt_term_buffer_ = 0;
        adapt_base_window_ = 0;

        restart();
      }

      void restart() {
        adapt_window_counter_ = 0;
        adapt_window_size_ = adapt_base_window_;
        adapt_next_window_ = adapt_init_buffer_ + adapt_window_size_ - 1;
      }

      void set_window_params(unsigned int num_warmup,
                             unsigned int init_buffer,
                             unsigned int term_buffer,
                             unsigned int base_window,
                             interface_callbacks::writer::base_writer& writer) {
        if (num_warmup < 20) {
          writer("WARNING: No " + estimator_name_ + " estimation is");
          writer("         performed for num_warmup < 20");
          writer();
          return;
        }

        if (init_buffer + base_window + term_buffer > num_warmup) {
          writer("WARNING: The initial buffer, adaptation window, "
                 "and terminal buffer");
          writer("         overflow the total number of warmup iterations.");

          num_warmup_ = num_warmup;
          adapt_init_buffer_ = 0.15 * num_warmup;
          adapt_term_buffer_ = 0.10 * num_warmup;
          adapt_base_window_
            = num_warmup - (adapt_init_buffer_ + adapt_term_buffer_);

          writer("         Defaulting to a 15%/75%/10% partition,");

          std::stringstream msg;
          msg << "           init_buffer = " << adapt_init_buffer_;
          writer(msg.str());

          msg.str("");
          msg << "           adapt_window = " << adapt_base_window_;
          writer(msg.str());

          msg.str("");
          msg << "           term_buffer = " << adapt_term_buffer_;
          writer(msg.str());

          writer();
          return;
        }

        num_warmup_ = num_warmup;
        adapt_init_buffer_ = init_buffer;
        adapt_term_buffer_ = term_buffer;
        adapt_base_window_ = base_window;
        restart();
      }

      bool adaptation_window() {
        return (adapt_window_counter_ >= adapt_init_buffer_)
               && (adapt_window_counter_ < num_warmup_ - adapt_term_buffer_)
               && (adapt_window_counter_ != num_warmup_);
      }

      bool end_adaptation_window() {
        return (adapt_window_counter_ == adapt_next_window_)
               && (adapt_window_counter_ != num_warmup_);
      }

      void compute_next_window() {
        if (adapt_next_window_ == num_warmup_ - adapt_term_buffer_ - 1)
          return;

        adapt_window_size_ *= 2;
        adapt_next_window_ = adapt_window_counter_ + adapt_window_size_;

        if (adapt_next_window_ == num_warmup_ - adapt_term_buffer_ - 1) return;

        // Bounday of the following window, not the window just computed
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

      unsigned int num_warmup_;
      unsigned int adapt_init_buffer_;
      unsigned int adapt_term_buffer_;
      unsigned int adapt_base_window_;

      unsigned int adapt_window_counter_;
      unsigned int adapt_next_window_;
      unsigned int adapt_window_size_;
    };

  }  // mcmc
}  // stan
#endif
