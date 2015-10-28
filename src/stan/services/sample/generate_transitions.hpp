#ifndef STAN_SERVICES_SAMPLE_GENERATE_TRANSITIONS_HPP
#define STAN_SERVICES_SAMPLE_GENERATE_TRANSITIONS_HPP

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/sample/mcmc_writer.hpp>
#include <stan/services/mcmc/print_progress.hpp>
#include <string>

namespace stan {
  namespace services {
    namespace sample {

      template <class Model, class RNG, class StartTransitionCallback,
                class SampleRecorder, class DiagnosticRecorder,
                class MessageRecorder>
      void generate_transitions(stan::mcmc::base_mcmc* sampler,
                                const int num_iterations,
                                const int start,
                                const int finish,
                                const int num_thin,
                                const int refresh,
                                const bool save,
                                const bool warmup,
                                stan::services::sample::mcmc_writer<
                                Model, SampleRecorder,
                                DiagnosticRecorder, MessageRecorder>& writer,
                                stan::mcmc::sample& init_s,
                                Model& model,
                                RNG& base_rng,
                                const std::string& prefix,
                                const std::string& suffix,
                                std::ostream& o,
                                StartTransitionCallback& callback) {
        for (int m = 0; m < num_iterations; ++m) {
          callback();

          mcmc::print_progress(m, start, finish, refresh, warmup, prefix, suffix, o);

          init_s = sampler->transition(init_s);

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
