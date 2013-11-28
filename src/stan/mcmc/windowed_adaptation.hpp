#ifndef __STAN__MCMC__WINDOWED__ADAPTATION__BETA__
#define __STAN__MCMC__WINDOWED__ADAPTATION__BETA__

#include <ostream>
#include <string>

#include <stan/mcmc/base_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
        
    class windowed_adaptation: public base_adaptation {
      
    public:
      
      windowed_adaptation(std::string name): _estimator_name(name) {
        _num_warmup = 0;
        _adapt_init_buffer = 0;
        _adapt_term_buffer = 0;
        _adapt_base_window = 0;
        
        restart();
      }
      
      void restart() {
        _adapt_window_counter = 0;
        _adapt_window_size = _adapt_base_window;
        _adapt_next_window = _adapt_init_buffer + _adapt_window_size - 1;
      }
      
      void set_window_params(unsigned int num_warmup,
                             unsigned int init_buffer,
                             unsigned int term_buffer,
                             unsigned int base_window,
                             std::ostream* e = 0) {
        
        if (num_warmup < 20) {
          if (e) {
            *e << "WARNING: No " << _estimator_name << " estimation is" << std::endl;
            *e << "         performed for num_warmup < 20" << std::endl << std::endl;
          }
          return;
        }
        
        if (init_buffer + base_window + term_buffer > num_warmup) {
          
          if (e) {
            *e << "WARNING: The initial buffer, adaptation window, and terminal buffer" << std::endl;
            *e << "         overflow the total number of warmup iterations." << std::endl;
          }
          
          _num_warmup = num_warmup;
          _adapt_init_buffer = 0.15 * num_warmup;
          _adapt_term_buffer = 0.10 * num_warmup;
          _adapt_base_window = num_warmup - (_adapt_init_buffer + _adapt_term_buffer);
          
          if(e) {
            *e << "         Defaulting to a 15%/75%/10% partition," << std::endl;
            *e << "           init_buffer = " << _adapt_init_buffer << std::endl;
            *e << "           adapt_window = " << _adapt_base_window << std::endl;
            *e << "           term_buffer = " << _adapt_term_buffer << std::endl << std::endl;
          }
          
          return;
          
        }
        
        _num_warmup = num_warmup;
        _adapt_init_buffer = init_buffer;
        _adapt_term_buffer = term_buffer;
        _adapt_base_window = base_window;
        restart();
        
      }
      
      bool adaptation_window() {
        return (_adapt_window_counter >= _adapt_init_buffer)
               && (_adapt_window_counter < _num_warmup - _adapt_term_buffer)
               && (_adapt_window_counter != _num_warmup);
      }
      
      bool end_adaptation_window() {
        return (_adapt_window_counter == _adapt_next_window)
               && (_adapt_window_counter != _num_warmup);
      }
      
      void compute_next_window() {
        
        if (_adapt_next_window == _num_warmup - _adapt_term_buffer - 1) return;
        
        _adapt_window_size *= 2;
        _adapt_next_window = _adapt_window_counter + _adapt_window_size;
        
        if (_adapt_next_window == _num_warmup - _adapt_term_buffer - 1) return;
        
        // Bounday of the following window, not the window just computed
        unsigned int next_window_boundary = _adapt_next_window + 2 * _adapt_window_size;
        
        // If the following window overtakes the full adaptation window,
        // then stretch the current window to the end of the full window
        if (next_window_boundary >= _num_warmup - _adapt_term_buffer) {
          _adapt_next_window = _num_warmup - _adapt_term_buffer - 1;
        }
        
      }
      
    protected:
      
      std::string _estimator_name;
      
      unsigned int _num_warmup;
      unsigned int _adapt_init_buffer;
      unsigned int _adapt_term_buffer;
      unsigned int _adapt_base_window;
      
      unsigned int _adapt_window_counter;
      unsigned int _adapt_next_window;
      unsigned int _adapt_window_size;
      
    };
    
  } // mcmc
  
} // stan

#endif
