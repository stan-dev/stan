#ifndef STAN_OLD_SERVICES_MCMC_WARMUP_HPP
#define STAN_OLD_SERVICES_MCMC_WARMUP_HPP

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/callbacks/interrupt/base_interrupt.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <stan/services/util/generate_transitions.hpp>
#include <string>

namespace stan {
  namespace services {
    namespace mcmc {

      template <class Model, class RNG>
      void warmup(stan::mcmc::base_mcmc& sampler,
                  int num_warmup,
                  int num_samples,
                  int num_thin,
                  int refresh,
                  bool save,
                  stan::services::sample::mcmc_writer&
                  mcmc_writer,
                  stan::mcmc::sample& init_s,
                  Model& model,
                  RNG& base_rng,
                  stan::callbacks::interrupt::base_interrupt&
                  callback,
                  callbacks::writer::base_writer& info_writer,
                  callbacks::writer::base_writer& error_writer) {
        util::generate_transitions<Model, RNG>
          (sampler, num_warmup, 0, num_warmup + num_samples, num_thin,
           refresh, save, true,
           mcmc_writer,
           init_s, model, base_rng,
           callback, info_writer, error_writer);
      }

    }
  }
}

#endif
