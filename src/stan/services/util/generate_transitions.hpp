#ifndef STAN_SERVICES_UTIL_GENERATE_TRANSITIONS_HPP
#define STAN_SERVICES_UTIL_GENERATE_TRANSITIONS_HPP

#include <stan/callbacks/writer/base_writer.hpp>
#include <stan/callbacks/interrupt/base_interrupt.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <string>

namespace stan {
  namespace services {
    namespace util {

      template <class Model, class RNG>
      void generate_transitions(stan::mcmc::base_mcmc& sampler,
                                const int num_iterations,
                                const int start,
                                const int finish,
                                const int num_thin,
                                const int refresh,
                                const bool save,
                                const bool warmup,
                                stan::services::sample::mcmc_writer&
                                mcmc_writer,
                                stan::mcmc::sample& init_s,
                                Model& model,
                                RNG& base_rng,
                           stan::callbacks::interrupt::base_interrupt&
                                callback,
                                callbacks::writer::base_writer&
                                info_writer,
                                callbacks::writer::base_writer&
                                error_writer) {
        for (int m = 0; m < num_iterations; ++m) {
          callback();

          if (refresh > 0
              && (start + m + 1 == finish
                  || m == 0
                  || (m + 1) % refresh == 0)) {
            int it_print_width
              = std::ceil(std::log10(static_cast<double>(finish)));
            std::stringstream message;
            message << "Iteration: ";
            message << std::setw(it_print_width) << m + 1 + start
                    << " / " << finish;
            message << " [" << std::setw(3)
                    << static_cast<int>( (100.0 * (start + m + 1)) / finish )
                    << "%] ";
            message << (warmup ? " (Warmup)" : " (Sampling)");

            info_writer(message.str());
          }

          init_s = sampler.transition(init_s, info_writer, error_writer);

          if ( save && ( (m % num_thin) == 0) ) {
            mcmc_writer.write_sample_params(base_rng, init_s, sampler, model);
            mcmc_writer.write_diagnostic_params(init_s, sampler);
          }
        }
      }

    }
  }
}

#endif
