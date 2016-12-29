#ifndef STAN_SERVICES_UTIL_GENERATE_TRANSITIONS_HPP
#define STAN_SERVICES_UTIL_GENERATE_TRANSITIONS_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <string>

namespace stan {
  namespace services {
    namespace util {

      /**
       * Generates MCMC transitions.
       *
       * @tparam Model model class
       * @tparam RNG random number generator class
       * @param sampler MCMC sampler used to generate transitions
       * @param num_iterations number of MCMC transitions
       * @param start starting iteration number used for printing messages
       * @param finish end iteration number used for printing messages
       * @param num_thin when save is true, a draw will be written to the
       *   mcmc_writer every num_thin iterations
       * @param refresh number of iterations to print a message. If
       *   refresh is zero, iteration number messages will not be printed
       * @param save if save is true, the transitions will be written
       *   to the mcmc_writer. If false, transitions will not be written
       * @param warmup indicates whether these transitions are warmup. Used
       *   for printing iteration number messages
       * @param mcmc_writer writer to handle mcmc otuput
       * @param[in,out] init_s starts as the initial unconstrained parameter
       *   values. When the function completes, this will have the final iteration's
       *   unconstrained parameter values
       * @param model model
       * @param base_rng random number generator
       * @param callback interrupt callback called once an iteration
       * @param info_writer writer for informational messages
       * @param error_writer writer for error messages
       */
      template <class Model, class RNG>
      void generate_transitions(stan::mcmc::base_mcmc& sampler,
                                const int num_iterations,
                                const int start,
                                const int finish,
                                const int num_thin,
                                const int refresh,
                                const bool save,
                                const bool warmup,
                                util::mcmc_writer& mcmc_writer,
                                stan::mcmc::sample& init_s,
                                Model& model,
                                RNG& base_rng,
                                callbacks::interrupt& callback,
                                callbacks::writer& info_writer,
                                callbacks::writer& error_writer) {
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

          if (save && ((m % num_thin) == 0)) {
            mcmc_writer.write_sample_params(base_rng, init_s, sampler, model);
            mcmc_writer.write_diagnostic_params(init_s, sampler);
          }
        }
      }

    }
  }
}

#endif
