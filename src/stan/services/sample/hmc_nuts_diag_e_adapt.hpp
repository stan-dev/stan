#ifndef STAN_SERVICES_SAMPLE_HMC_NUTS_DIAG_E_ADAPT_HPP
#define STAN_SERVICES_SAMPLE_HMC_NUTS_DIAG_E_ADAPT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/services/util/run_adaptive_sampler.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/initialize.hpp>
#include <vector>

namespace stan {
  namespace services {
    namespace sample {

      /**
       * Runs HMC with NUTS with diagonal Euclidean
       * metric with adaptation.
       *
       * @tparam Model Model class
       * @param[in] model Input model to test (with data already instantiated)
       * @param[in] init var context for initialization
       * @param[in] random_seed random seed for the random number generator
       * @param[in] chain chain id to advance the pseudo random number generator
       * @param[in] init_radius radius to initialize
       * @param[in] num_warmup Number of warmup samples
       * @param[in] num_samples Number of samples
       * @param[in] num_thin Number to thin the samples
       * @param[in] save_warmup Indicates whether to save the warmup iterations
       * @param[in] refresh Controls the output
       * @param[in] stepsize initial stepsize for discrete evolution
       * @param[in] stepsize_jitter uniform random jitter of stepsize
       * @param[in] max_depth Maximum tree depth
       * @param[in] delta adaptation target acceptance statistic
       * @param[in] gamma adaptation regularization scale
       * @param[in] kappa adaptation relaxation exponent
       * @param[in] t0 adaptation iteration offset
       * @param[in] init_buffer width of initial fast adaptation interval
       * @param[in] term_buffer width of final fast adaptation interval
       * @param[in] window initial width of slow adaptation interval
       * @param[in,out] interrupt Callback for interrupts
       * @param[in,out] message_writer Writer for messages
       * @param[in,out] error_writer Writer for errors
       * @param[in,out] init_writer Writer callback for unconstrained inits
       * @param[in,out] sample_writer Writer for draws
       * @param[in,out] diagnostic_writer Writer for diagnostic information
       * @return error_codes::OK if successful
       */
      template <class Model>
      int hmc_nuts_diag_e_adapt(Model& model, stan::io::var_context& init,
                                unsigned int random_seed, unsigned int chain,
                                double init_radius, int num_warmup,
                                int num_samples, int num_thin, bool save_warmup,
                                int refresh, double stepsize,
                                double stepsize_jitter, int max_depth,
                                double delta, double gamma, double kappa,
                                double t0, unsigned int init_buffer,
                                unsigned int term_buffer, unsigned int window,
                                callbacks::interrupt& interrupt,
                                callbacks::writer& message_writer,
                                callbacks::writer& error_writer,
                                callbacks::writer& init_writer,
                                callbacks::writer& sample_writer,
                                callbacks::writer& diagnostic_writer) {
        boost::ecuyer1988 rng = util::create_rng(random_seed, chain);

        std::vector<int> disc_vector;
        std::vector<double> cont_vector
          = util::initialize(model, init, rng, init_radius, true,
                             message_writer, init_writer);

        stan::mcmc::adapt_diag_e_nuts<Model, boost::ecuyer1988>
          sampler(model, rng);
        sampler.set_nominal_stepsize(stepsize);
        sampler.set_stepsize_jitter(stepsize_jitter);
        sampler.set_max_depth(max_depth);

        sampler.get_stepsize_adaptation().set_mu(log(10 * stepsize));
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_kappa(kappa);
        sampler.get_stepsize_adaptation().set_t0(t0);

        sampler.set_window_params(num_warmup, init_buffer, term_buffer,
                                  window, message_writer);

        util::run_adaptive_sampler(sampler, model, cont_vector, num_warmup,
                                   num_samples, num_thin, refresh, save_warmup,
                                   rng, interrupt, message_writer, error_writer,
                                   sample_writer, diagnostic_writer);

        return error_codes::OK;
      }

    }
  }
}
#endif
