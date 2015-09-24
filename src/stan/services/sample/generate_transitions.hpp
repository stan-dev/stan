#ifndef STAN_SERVICES_SAMPLE_GENERATE_TRANSITIONS_HPP
#define STAN_SERVICES_SAMPLE_GENERATE_TRANSITIONS_HPP

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/io/do_print.hpp>
#include <stan/services/sample/progress.hpp>
#include <string>

namespace stan {
  namespace services {
    namespace sample {
      /**
       * @tparam Sampler MCMC sampler implementation
       * @tparam MCMCWriter MCMC writer implementation
       * @tparam Interrupt Interrupt callback implementation
       * @param sampler MCMC sampler
       * @param sample Initial sample
       * @param num_iterations Number of iterations
       * @param start Initial iteration index
       * @param finish Final iteration index
       * @param num_thin Thinning stride
       * @param refresh Progress update rate
       * @param save Flag to save samples
       * @param warmup Flag to indicate warmup
       * @param mcmc_writer MCMC writer
       * @param interrupt Interrupt callback called at the beginning
       of each iteration
       */
      template <class Sampler, class MCMCWriter, class Interrupt>
      void generate_transitions(Sampler& sampler,
                                stan::mcmc::sample& sample,
                                int num_iterations,
                                int start,
                                int finish,
                                int num_thin,
                                int refresh,
                                bool save,
                                bool warmup,
                                MCMCWriter& mcmc_writer,
                                Interrupt& interrupt) {
        for (int n = 0; n < num_iterations; ++n) {
          interrupt();

          if (io::do_print(n, (start + n + 1 == finish), refresh)) {
            std::string msg = progress(n, start, finish, refresh, warmup);
            mcmc_writer.write_info_message(msg);
          }

          sample = sampler.transition(sample);
          mcmc_writer.write_info_message(sampler.flush_info_buffer());
          mcmc_writer.write_err_message(sampler.flush_err_buffer());
          sampler.clear_buffers();

          if ( save && ( (n % num_thin) == 0) )
            mcmc_writer.write_state(sample, sampler);
        }
      }

    }  // sample
  }  // services
}  // stan

#endif
