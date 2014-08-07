#ifndef STAN__GM__ARGUMENTS__ADAPT__HPP
#define STAN__GM__ARGUMENTS__ADAPT__HPP

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_adapt_engaged.hpp>
#include <stan/gm/arguments/arg_adapt_gamma.hpp>
#include <stan/gm/arguments/arg_adapt_delta.hpp>
#include <stan/gm/arguments/arg_adapt_kappa.hpp>
#include <stan/gm/arguments/arg_adapt_t0.hpp>
#include <stan/gm/arguments/arg_adapt_init_buffer.hpp>
#include <stan/gm/arguments/arg_adapt_term_buffer.hpp>
#include <stan/gm/arguments/arg_adapt_window.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_adapt: public categorical_argument {
      
    public:
      
      arg_adapt() {
        
        _name = "adapt";
        _description = "Warmup Adaptation";
        
        _subarguments.push_back(new arg_adapt_engaged());
        _subarguments.push_back(new arg_adapt_gamma());
        _subarguments.push_back(new arg_adapt_delta());
        _subarguments.push_back(new arg_adapt_kappa());
        _subarguments.push_back(new arg_adapt_t0());
        _subarguments.push_back(new arg_adapt_init_buffer());
        _subarguments.push_back(new arg_adapt_term_buffer());
        _subarguments.push_back(new arg_adapt_window());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

