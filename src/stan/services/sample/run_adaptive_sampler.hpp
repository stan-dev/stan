#ifndef STAN_SERVICES_SAMPLE_RUN_ADAPTIVE_SAMPLER_HPP
#define STAN_SERVICES_SAMPLE_RUN_ADAPTIVE_SAMPLER_HPP

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/sample/generate_transitions.hpp>

namespace stan {
  namespace services {
    namespace sample {
      
      /**
       * @tparam Sampler MCMC sampler implementation
       * @tparam MCMCWriter MCMC writer implementation
       * @tparam Interrupt Interrupt callback implementation
       * @param sampler MCMC sampler
       * @param sample Initial sample
       * @param num_warmup Number of warmup iterations
       * @param num_samples Number of sampling iterations
       * @param refresh Progress update rate
       * @param save_warmup Flag to save warmup iterations
       * @param writer MCMC writer
       * @param iteration_interrupt Interrupt callback called at the beginning
       of each iteration
       */
      template <class Sampler, class MCMCWriter, class Interrupt>
      void run_adaptive_sampler(Sampler& sampler,
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
        generate_transitions(sampler, sample,
                             num_warmup, 0, num_warmup, num_thin,
                             refresh, save_warmup, true, writer, interrupt);
        clock_t end = clock();
        warm_delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        
        sampler.disengage_adaptation();
        writer.write_adapt_finish(sampler);
        
        // Sampling
        start = clock();
        generate_transitions(sampler, sample,
                             num_samples, num_warmup, num_warmup + num_samples, num_thin,
                             refresh, true, false, writer, interrupt);
        end = clock();
        sample_delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        
        writer.write_timing(warm_delta_t, sample_delta_t);

      }

    } // sample
  } // services
} // stan

#endif
