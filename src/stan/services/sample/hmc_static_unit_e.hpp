#ifndef STAN_SERVICES_SAMPLE_HMC_STATIC_UNIT_E_HPP
#define STAN_SERVICES_SAMPLE_HMC_STATIC_UNIT_E_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/interface_callbacks/interrupt/base_interrupt.hpp>
#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/check_timing.hpp>
#include <stan/services/sample/run_sampler.hpp>
#include <stan/services/mcmc/sample.hpp>
#include <stan/services/mcmc/warmup.hpp>
#include <stan/mcmc/hmc/static/unit_e_static_hmc.hpp>
#include <ctime>

namespace stan {
  namespace services {
    namespace sample {

      /**
       * Runs HMC for static integration time with unit Euclidean
       * metric without adapatation.
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
      int hmc_static_unit_e(Model& model,
                            rng_t& base_rng,
                            Eigen::VectorXd& cont_params,
                            int num_warmup,
                            int num_samples,
                            int num_thin,
                            bool save_warmup,
                            int refresh,
                            double epsilon,
                            double epsilon_jitter,
                            double int_time,
                            interface_callbacks::interrupt::base_interrupt& interrupt,
                            interface_callbacks::writer::base_writer& sample_writer,
                            interface_callbacks::writer::base_writer& diagnostic_writer,
                            interface_callbacks::writer::base_writer& message_writer) {
        stan::services::check_timing(model, cont_params, message_writer);

        stan::mcmc::unit_e_static_hmc<Model, rng_t> sampler(model, base_rng);
        sampler.set_nominal_stepsize_and_T(epsilon, int_time);
        sampler.set_stepsize_jitter(epsilon_jitter);

        run_sampler(sampler, model,
                    cont_params,
                    num_warmup, num_samples, num_thin,
                    refresh, save_warmup, false, base_rng,
                    interrupt, sample_writer, diagnostic_writer,
                    message_writer);
        
        return stan::services::error_codes::OK;
      }
      
    }
  }
}
#endif
