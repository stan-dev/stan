#ifndef STAN_SERVICES_MCMC_RUN_MARKOV_CHAIN_HPP
#define STAN_SERVICES_MCMC_RUN_MARKOV_CHAIN_HPP

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/io/mcmc_writer.hpp>
#include <stan/services/mcmc/print_progress.hpp>
#include <string>

namespace stan {
  namespace services {
    namespace mcmc {

      template <class Model, class RNG, class StartTransitionCallback,
                class SampleRecorder, class DiagnosticRecorder,
                class MessageRecorder>
      void run_markov_chain(stan::mcmc::base_mcmc* sampler,
                            const int num_iterations,
                            const int start,
                            const int finish,
                            const int num_thin,
                            const int refresh,
                            const bool save,
                            const bool warmup,
                            stan::io::mcmc_writer<
                               Model, SampleRecorder,
                               DiagnosticRecorder, MessageRecorder>& writer,
                            stan::mcmc::sample& init_s,
                            Model& model,
                            RNG& base_rng,
                            const std::string& prefix,
                            const std::string& suffix,
                            std::ostream& o,
                            std::ostream* timing_stream,
                            StartTransitionCallback& callback) {
        // Timing variables
        clock_t start_time = clock();
        clock_t end_time;
        double delta_t;

        for (int m = 0; m < num_iterations; ++m) {
          callback();

          print_progress(m, start, finish, refresh, warmup, prefix, suffix, o);

          init_s = sampler->transition(init_s);

          end_time = clock();
          delta_t = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
          *timing_stream << m;
          *timing_stream << ", ";
          *timing_stream << delta_t;
          *timing_stream << std::endl;

          if ( save && ( (m % num_thin) == 0) ) {
            writer.write_sample_params(base_rng, init_s, *sampler, model);
            writer.write_diagnostic_params(init_s, sampler);
          }
        }
      }

    }
  }
}

#endif
