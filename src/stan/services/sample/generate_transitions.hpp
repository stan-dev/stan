#ifndef STAN_SERVICES_SAMPLE_GENERATE_TRANSITIONS_HPP
#define STAN_SERVICES_SAMPLE_GENERATE_TRANSITIONS_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/sample/mcmc_writer.hpp>
#include <stan/services/sample/progress.hpp>
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
                                DiagnosticRecorder, MessageRecorder>&
                                mcmc_writer,
                                stan::mcmc::sample& init_s,
                                Model& model,
                                RNG& base_rng,
                                const std::string& prefix,
                                const std::string& suffix,
                                std::ostream& o,
                                StartTransitionCallback& callback,
                                interface_callbacks::writer::base_writer&
                                info_writer,
                                interface_callbacks::writer::base_writer&
                                error_writer) {
        for (int m = 0; m < num_iterations; ++m) {
          callback();

          progress(m, start, finish, refresh, warmup, prefix, suffix, o);

          init_s = sampler->transition(init_s, info_writer, error_writer);

          if ( save && ( (m % num_thin) == 0) ) {
            mcmc_writer.write_sample_params(base_rng, init_s, *sampler, model);
            mcmc_writer.write_diagnostic_params(init_s, sampler);
          }
        }
      }

    }
  }
}

#endif
