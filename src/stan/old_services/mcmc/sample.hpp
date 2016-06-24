#ifndef STAN_OLD_SERVICES_MCMC_SAMPLE_HPP
#define STAN_OLD_SERVICES_MCMC_SAMPLE_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/interface_callbacks/interrupt/base_interrupt.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/old_services/sample/mcmc_writer.hpp>
#include <stan/old_services/sample/generate_transitions.hpp>
#include <string>

namespace stan {
  namespace services {
    namespace mcmc {

      template <class Model, class RNG>
      void sample(stan::mcmc::base_mcmc& sampler,
                  int num_warmup,
                  int num_samples,
                  int num_thin,
                  int refresh,
                  bool save,
                  stan::services::sample::mcmc_writer<Model>&
                  mcmc_writer,
                  stan::mcmc::sample& init_s,
                  Model& model,
                  RNG& base_rng,
                  interface_callbacks::interrupt::base_interrupt& callback,
                  interface_callbacks::writer::base_writer& info_writer,
                  interface_callbacks::writer::base_writer& error_writer) {
        stan::services::sample::generate_transitions<Model, RNG>
          (sampler, num_samples, num_warmup, num_warmup + num_samples, num_thin,
           refresh, save, false,
           mcmc_writer,
           init_s, model, base_rng,
           callback, info_writer, error_writer);
      }

    }
  }
}

#endif
