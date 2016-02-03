#ifndef STAN_SERVICES_SAMPLE_HMC_NUTS_UNIT_E_HPP
#define STAN_SERVICES_SAMPLE_HMC_NUTS_UNIT_E_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/interface_callbacks/interrupt/base_interrupt.hpp>
#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/check_timing.hpp>
#include <stan/services/sample/run_sampler.hpp>
#include <stan/mcmc/hmc/nuts/unit_e_nuts.hpp>
#include <ctime>

namespace stan {
  namespace services {
    namespace sample {

      /**
       * Runs HMC with NUTS with unit Euclidean
       * metric without adapatation.
       *
       * @tparam Model Model class
       * @tparam rng_t Random number generator class
       * @param model Instance of model
       * @param base_rng Instance of random number generator
       * @param cont_params Initial value
       * @param num_warmup Number of warmup samples
       * @param num_samples Number of samples
       * @param num_thin Number to thin the samples
       * @param save_warmup Indicates whether to save the warmup iterations
       * @param refresh Controls the output
       * @param stepsize initial stepsize for discrete evolution
       * @param stepsize_jitter uniform random jitter of stepsize
       * @param max_depth Maximum tree depth
       * @param interrupt Callback for interrupts
       * @param sample_writer Writer for draws
       * @param diagnostic_writer Writer for diagnostic information
       * @param message_writer Writer for messages
       * @return error code; 0 if no error
       */
      template <class Model, class rng_t>
      int hmc_nuts_unit_e(Model& model,
                          rng_t& base_rng,
                          Eigen::VectorXd& cont_params,
                          int num_warmup,
                          int num_samples,
                          int num_thin,
                          bool save_warmup,
                          int refresh,
                          double stepsize,
                          double stepsize_jitter,
                          int max_depth,
                interface_callbacks::interrupt::base_interrupt& interrupt,
                interface_callbacks::writer::base_writer& sample_writer,
                interface_callbacks::writer::base_writer& diagnostic_writer,
                interface_callbacks::writer::base_writer& message_writer) {
        stan::services::check_timing(model, cont_params, message_writer);

        stan::mcmc::unit_e_nuts<Model, rng_t> sampler(model, base_rng);
        sampler.set_nominal_stepsize(stepsize);
        sampler.set_stepsize_jitter(stepsize_jitter);
        sampler.set_max_depth(max_depth);

        run_sampler(sampler, model,
                    cont_params,
                    num_warmup, num_samples, num_thin,
                    refresh, save_warmup, base_rng,
                    interrupt, sample_writer, diagnostic_writer,
                    message_writer);

        return stan::services::error_codes::OK;
      }

    }
  }
}
#endif
