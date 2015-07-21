#ifndef STAN_SERVICES_ARGUMENTS_ARG_SAMPLE_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_SAMPLE_HPP

#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/arg_num_samples.hpp>
#include <stan/services/arguments/arg_num_warmup.hpp>
#include <stan/services/arguments/arg_save_warmup.hpp>
#include <stan/services/arguments/arg_thin.hpp>
#include <stan/services/arguments/arg_adapt.hpp>
#include <stan/services/arguments/arg_sample_algo.hpp>

namespace stan {
  namespace services {

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

  }  // services
}  // stan

#endif

