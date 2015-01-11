#ifndef STAN__MCMC__WINDOWED__ADAPTATION__BETA
#define STAN__MCMC__WINDOWED__ADAPTATION__BETA

#include <ostream>
#include <string>

#include <stan/mcmc/base_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
        
    class windowed_adaptation: public base_adaptation {
      
    public:
      
      windowed_adaptation(std::string name): estimator_name_(name) {
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
                             std::ostream* e = 0) {
        
        if (num_warmup < 20) {
          if (e) {
            *e << "WARNING: No " << estimator_name_ << " estimation is" << std::endl;
            *e << "         performed for num_warmup < 20" << std::endl << std::endl;
          }
          return;
        }
        
        if (init_buffer + base_window + term_buffer > num_warmup) {
          
          if (e) {
            *e << "WARNING: The initial buffer, adaptation window, and terminal buffer" << std::endl;
            *e << "         overflow the total number of warmup iterations." << std::endl;
          }
          
          num_warmup_ = num_warmup;
          adapt_init_buffer_ = 0.15 * num_warmup;
          adapt_term_buffer_ = 0.10 * num_warmup;
          adapt_base_window_ = num_warmup - (adapt_init_buffer_ + adapt_term_buffer_);
          
          if(e) {
            *e << "         Defaulting to a 15%/75%/10% partition," << std::endl;
            *e << "           init_buffer = " << adapt_init_buffer_ << std::endl;
            *e << "           adapt_window = " << adapt_base_window_ << std::endl;
            *e << "           term_buffer = " << adapt_term_buffer_ << std::endl << std::endl;
          }
          
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
        
        if (adapt_next_window_ == num_warmup_ - adapt_term_buffer_ - 1) return;
        
        adapt_window_size_ *= 2;
        adapt_next_window_ = adapt_window_counter_ + adapt_window_size_;
        
        if (adapt_next_window_ == num_warmup_ - adapt_term_buffer_ - 1) return;
        
        // Bounday of the following window, not the window just computed
        unsigned int next_window_boundary = adapt_next_window_ + 2 * adapt_window_size_;
        
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
    
  } // mcmc
  
} // stan

#endif
