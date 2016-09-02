#ifndef STAN_SERVICES_MCMC_SAMPLE_HPP
#define STAN_SERVICES_MCMC_SAMPLE_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/sample/mcmc_writer.hpp>
#include <stan/services/sample/generate_transitions.hpp>
#include <string>

namespace stan {
  namespace services {
    namespace mcmc {

      template <class Model, class RNG, class StartTransitionCallback,
                class SampleRecorder, class DiagnosticRecorder,
                class MessageRecorder>
      void sample(stan::mcmc::base_mcmc* sampler,
                  int num_warmup,
                  int num_samples,
                  int num_thin,
                  int refresh,
                  bool save,
                  stan::services::sample::mcmc_writer<
                  Model, SampleRecorder, DiagnosticRecorder, MessageRecorder>&
                  mcmc_writer,
                  stan::mcmc::sample& init_s,
                  Model& model,
                  RNG& base_rng,
                  const std::string& prefix,
                  const std::string& suffix,
                  std::ostream& o,
                  StartTransitionCallback& callback,
                  interface_callbacks::writer::base_writer& info_writer,
                  interface_callbacks::writer::base_writer& error_writer) {
        stan::services::sample::generate_transitions<Model, RNG,
                                                     StartTransitionCallback,
                                                     SampleRecorder,
                                                     DiagnosticRecorder,
                                                     MessageRecorder>
          (sampler, num_samples, num_warmup, num_warmup + num_samples, num_thin,
           refresh, save, false,
           mcmc_writer,
           init_s, model, base_rng,
           prefix, suffix, o,
           callback, info_writer, error_writer);
      }

    }
  }
}

#endif
