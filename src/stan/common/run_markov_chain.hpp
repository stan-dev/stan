#ifndef __STAN__COMMON__RUN_MARKOV_CHAIN_HPP__
#define __STAN__COMMON__RUN_MARKOV_CHAIN_HPP__

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/io/mcmc_writer.hpp>
#include <stan/common/print_progress.hpp>

namespace stan {
  namespace common {

    template <class Model, class RNG>
    void run_markov_chain(stan::mcmc::base_mcmc* sampler,
                          const int num_iterations,
                          const int start,
                          const int finish,
                          const int num_thin,
                          const int refresh,
                          const bool save,
                          const bool warmup,
                          stan::io::mcmc_writer<Model>& writer,
                          stan::mcmc::sample& init_s,
                          Model& model,
                          RNG& base_rng,
                          const std::string& prefix,
                          const std::string& suffix,
                          std::ostream& o) {

      for (int m = 0; m < num_iterations; ++m) {
      
        print_progress(m, start, finish, refresh, warmup, prefix, suffix, o);
      
        init_s = sampler->transition(init_s);
          
        if ( save && ( (m % num_thin) == 0) ) {
          writer.print_sample_params(base_rng, init_s, *sampler, model);
          writer.print_diagnostic_params(init_s, sampler);
        }

      }
      
    }

  }
}

#endif
