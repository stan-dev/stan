#ifndef STAN_SERVICES_MCMC_RB_SAMPLE_HPP
#define STAN_SERVICES_MCMC_RB_SAMPLE_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/sample/mcmc_writer.hpp>
#include <stan/services/sample/progress.hpp>
#include <string>

namespace stan {
  namespace services {
    namespace mcmc {

      template <class Model, class RNG, class StartTransitionCallback,
                class SampleRecorder, class DiagnosticRecorder,
                class MessageRecorder>
      void rb_sample(stan::mcmc::base_mcmc* sampler,
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
        for (int m = 0; m < num_samples; ++m) {
          callback();

          sample::progress(m, num_warmup, num_warmup + num_samples,
                           refresh, 0, prefix, suffix, o);

          std::vector<stan::mcmc::sample> rb_samples;

          init_s = sampler->rb_transition(init_s, rb_samples,
                                          info_writer, error_writer);

          if ( save && ( (m % num_thin) == 0) ) {
            for (size_t n = 0; n < rb_samples.size(); ++n)
              mcmc_writer.write_sample_params(base_rng, rb_samples.at(n),
                                              *sampler, model);
            mcmc_writer.sample_newline();

            mcmc_writer.write_diagnostic_params(init_s, sampler);
          }

        }
      }
    }  // mcmc
  }  // services
} // stan

#endif
