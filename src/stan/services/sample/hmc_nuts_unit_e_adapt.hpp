#ifndef STAN_SERVICES_SAMPLE_HMC_NUTS_UNIT_E_ADAPT_HPP
#define STAN_SERVICES_SAMPLE_HMC_NUTS_UNIT_E_ADAPT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/services/util/run_adaptive_sampler.hpp>
#include <stan/services/util/rng.hpp>
#include <stan/services/util/initialize.hpp>
#include <vector>

namespace stan {
  namespace services {
    namespace sample {

      /**
       * Runs HMC with NUTS with unit Euclidean
       * metric with adaptation.
       *
       * @tparam Model Model class
       *
       * @param model Input model to test (with data already instantiated)
       * @param init var context for initialization
       * @param random_seed random seed for the pseudo random number generator
       * @param chain chain id to advance the pseudo random number generator
       * @param init_radius radius to initialize
       * @param num_warmup Number of warmup samples
       * @param num_samples Number of samples
       * @param num_thin Number to thin the samples
       * @param save_warmup Indicates whether to save the warmup iterations
       * @param refresh Controls the output
       * @param stepsize initial stepsize for discrete evolution
       * @param stepsize_jitter uniform random jitter of stepsize
       * @param max_depth Maximum tree depth
       * @param delta adaptation target acceptance statistic
       * @param gamma adaptation regularization scale
       * @param kappa adaptation relaxation exponent
       * @param t0 adaptation iteration offset
       * @param interrupt Callback for interrupts
       * @param message_writer Writer for messages
       * @param error_writer Writer for errors
       * @param init_writer Writer callback for unconstrained inits
       * @param sample_writer Writer for draws
       * @param diagnostic_writer Writer for diagnostic information
       * @return stan::services::error_codes::OK if successful
       */
      template <class Model>
      int hmc_nuts_unit_e_adapt(Model& model,
                                stan::io::var_context& init,
                                unsigned int random_seed,
                                unsigned int chain,
                                double init_radius,
                                int num_warmup,
                                int num_samples,
                                int num_thin,
                                bool save_warmup,
                                int refresh,
                                double stepsize,
                                double stepsize_jitter,
                                int max_depth,
                                double delta,
                                double gamma,
                                double kappa,
                                double t0,
                                callbacks::interrupt&
                                interrupt,
                                callbacks::writer&
                                message_writer,
                                callbacks::writer&
                                error_writer,
                                callbacks::writer&
                                init_writer,
                                callbacks::writer&
                                sample_writer,
                                callbacks::writer&
                                diagnostic_writer) {
        boost::ecuyer1988 rng = stan::services::util::rng(random_seed, chain);

        std::vector<int> disc_vector;
        std::vector<double> cont_vector
          = stan::services::util::initialize(model, init, rng, init_radius,
                                             true,
                                             message_writer, init_writer);

        stan::mcmc::adapt_unit_e_nuts<Model, boost::ecuyer1988>
          sampler(model, rng);
        sampler.set_nominal_stepsize(stepsize);
        sampler.set_stepsize_jitter(stepsize_jitter);
        sampler.set_max_depth(max_depth);


        sampler.get_stepsize_adaptation().set_mu(log(10 * stepsize));
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_kappa(kappa);
        sampler.get_stepsize_adaptation().set_t0(t0);

        stan::services::util::run_adaptive_sampler(sampler, model,
                                                   cont_vector,
                                                   num_warmup, num_samples,
                                                   num_thin,
                                                   refresh, save_warmup, rng,
                                                   interrupt,
                                                   message_writer, error_writer,
                                                   sample_writer,
                                                   diagnostic_writer);

        return stan::services::error_codes::OK;
      }

    }
  }
}
#endif
