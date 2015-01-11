#ifndef STAN__COMMON__INIT_WINDOWED_ADAPT_HPP
#define STAN__COMMON__INIT_WINDOWED_ADAPT_HPP

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/gm/arguments/categorical_argument.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/common/init_adapt.hpp>

namespace stan {
  namespace common {
    
    template<class Sampler>
    bool init_windowed_adapt(stan::mcmc::base_mcmc* sampler, 
                             stan::gm::categorical_argument* adapt, 
                             unsigned int num_warmup, 
                             const Eigen::VectorXd& cont_params) {
      
      init_adapt<Sampler>(sampler, adapt, cont_params);
      
      unsigned int init_buffer 
        = dynamic_cast<stan::gm::u_int_argument*>(adapt->arg("init_buffer"))->value();
      unsigned int term_buffer 
        = dynamic_cast<stan::gm::u_int_argument*>(adapt->arg("term_buffer"))->value();
      unsigned int window = dynamic_cast<stan::gm::u_int_argument*>(adapt->arg("window"))->value();
      
      dynamic_cast<Sampler*>(sampler)->set_window_params(num_warmup, init_buffer, 
                                                         term_buffer, window, &std::cout);
      
      return true;
      
    }
    
  }
}

#endif
