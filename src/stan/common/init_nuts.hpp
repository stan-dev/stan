#ifndef STAN__COMMON__INIT_NUTS_HPP
#define STAN__COMMON__INIT_NUTS_HPP

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/gm/arguments/argument.hpp>
#include <stan/gm/arguments/categorical_argument.hpp>
#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  namespace common {
    
    template<class Sampler>
    bool init_nuts(stan::mcmc::base_mcmc* sampler, 
                   stan::gm::argument* algorithm) {
      stan::gm::categorical_argument* hmc 
        = dynamic_cast<stan::gm::categorical_argument*>
        (algorithm->arg("hmc"));
      
      stan::gm::categorical_argument* base 
        = dynamic_cast<stan::gm::categorical_argument*>
        (algorithm->arg("hmc")->arg("engine")->arg("nuts"));

      double epsilon 
        = dynamic_cast<stan::gm::real_argument*>(hmc->arg("stepsize"))->value();
      double epsilon_jitter 
        = dynamic_cast<stan::gm::real_argument*>(hmc->arg("stepsize_jitter"))->value();
      int max_depth = dynamic_cast<stan::gm::int_argument*>(base->arg("max_depth"))->value();
      
      dynamic_cast<Sampler*>(sampler)->set_nominal_stepsize(epsilon);
      dynamic_cast<Sampler*>(sampler)->set_stepsize_jitter(epsilon_jitter);
      dynamic_cast<Sampler*>(sampler)->set_max_depth(max_depth);
      
      return true;
    }
    
  }
}

#endif
