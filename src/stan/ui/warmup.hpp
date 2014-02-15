#ifndef __STAN__UI__WARMUP_HPP__
#define __STAN__UI__WARMUP_HPP__

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/io/mcmc_writer.hpp>
#include <stan/ui/run_markov_chain.hpp>

namespace stan {
  namespace ui {

    template <class Model, class RNG>
    void warmup(stan::mcmc::base_mcmc* sampler,
                int num_warmup,
                int num_samples,
                int num_thin,
                int refresh,
                bool save,
                stan::io::mcmc_writer<Model>& writer,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      run_markov_chain<Model, RNG>(sampler, num_warmup, 0, num_warmup + num_samples, num_thin,
                                   refresh, save, true,
                                   writer,
                                   init_s, model, base_rng);
    }
  }
}

#endif
