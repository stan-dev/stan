#ifndef STAN_SERVICES_SAMPLE_FIXED_PARAM_HPP
#define STAN_SERVICES_SAMPLE_FIXED_PARAM_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/interface_callbacks/interrupt/base_interrupt.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/sample/mcmc_writer.hpp>
#include <stan/services/sample/generate_transitions.hpp>

namespace stan {
  namespace services {
    namespace sample {

      /**
       * Runs the fixed_param sampler.
       *
       * @tparam Model Model class
       * @tparam rng_t Random number generator class
       * @param model Instance of model
       * @param base_rng Instance of random number generator
       * @param cont_params Initial value
       * @param num_samples Number of samples
       * @param num_thin Number to thin the samples
       * @param refresh Controls the output
       * @param interrupt Callback for interrupts
       * @param sample_writer Writer for draws
       * @param diagnostic_writer Writer for diagnostic information
       * @param message_writer Writer for messages
       * @return error code; 0 if no error
       */
      template <class Model, class rng_t>
      int fixed_param(Model& model,
                      rng_t& base_rng,
                      Eigen::VectorXd& cont_params,
                      int num_samples,
                      int num_thin,
                      int refresh,
                      interface_callbacks::interrupt::base_interrupt& interrupt,
                      interface_callbacks::writer::base_writer& sample_writer,
                  interface_callbacks::writer::base_writer& diagnostic_writer,
                  interface_callbacks::writer::base_writer& message_writer) {
        stan::mcmc::fixed_param_sampler sampler;
        stan::services::sample::mcmc_writer<Model>
          writer(sample_writer, diagnostic_writer, message_writer);
        stan::mcmc::sample s(cont_params, 0, 0);

        // Headers
        writer.write_sample_names(s, sampler, model);
        writer.write_diagnostic_names(s, sampler, model);

        clock_t start = clock();

        stan::services::sample::generate_transitions
          (sampler, num_samples, 0, num_samples, num_thin,
           refresh, true, false,
           writer,
           s, model, base_rng,
           interrupt, message_writer);
        clock_t end = clock();

        double sampleDeltaT = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        writer.write_timing(0.0, sampleDeltaT);

        return stan::services::error_codes::OK;
      }

    }
  }
}
#endif
