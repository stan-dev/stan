#ifndef __STAN__GM__ARGUMENTS__SAMPLE__HPP__
#define __STAN__GM__ARGUMENTS__SAMPLE__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_num_samples.hpp>
#include <stan/gm/arguments/arg_num_warmup.hpp>
#include <stan/gm/arguments/arg_save_warmup.hpp>
#include <stan/gm/arguments/arg_thin.hpp>
#include <stan/gm/arguments/arg_adapt.hpp>
#include <stan/gm/arguments/arg_sample_algo.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_sample: public categorical_argument {
      
    public:
      
      arg_sample() {
        
        _name = "sample";
        _description = "Bayesian inference with Markov Chain Monte Carlo";
        
        _subarguments.push_back(new arg_num_samples());
        _subarguments.push_back(new arg_num_warmup());
        _subarguments.push_back(new arg_save_warmup());
        _subarguments.push_back(new arg_thin());
        _subarguments.push_back(new arg_adapt());
        _subarguments.push_back(new arg_sample_algo());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

