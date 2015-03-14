#ifndef STAN__SERVICES__MCMC__ADAPT_SAMPLE_HPP
#define STAN__SERVICES__MCMC__ADAPT_SAMPLE_HPP

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/mcmc/generate_transitions.hpp>

namespace stan {
  namespace services {
    namespace mcmc {
      
      template <class Sampler, class MCMCWriter, class Interrupt>
      void adapt_sample(Sampler& sampler,
                        stan::mcmc::sample& sample,
                        int num_warmup,
                        int num_samples,
                        int num_thin,
                        int refresh,
                        bool save_warmup,
                        MCMCWriter& writer,
                        Interrupt& interrupt) {
        
        double warm_delta_t;
        double sample_delta_t;
        
        // Headers
        writer.write_names(sample, sampler);
        
        // Warm-Up
        clock_t start = clock();
        mcmc::generate_transitions(sampler, sample,
                                   num_warmup, 0, num_warmup, num_thin,
                                   refresh, save_warmup, true, writer, interrupt);
        clock_t end = clock();
        warm_delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        
        sampler.disengage_adaptation();
        writer.write_adapt_finish(sampler);
        
        // Sampling
        start = clock();
        mcmc::generate_transitions(sampler, sample,
                                   num_samples, num_warmup, num_warmup + num_samples, num_thin,
                                   refresh, true, false, writer, interrupt);
        end = clock();
        sample_delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        
        writer.write_timing(warm_delta_t, sample_delta_t);

      }

    } // mcmc
  } // services
} // stan

#endif
