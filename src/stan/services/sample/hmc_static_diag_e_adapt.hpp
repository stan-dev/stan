#ifndef STAN_SERVICES_SAMPLE_HMC_STATIC_DIAG_E_ADAPT_HPP
#define STAN_SERVICES_SAMPLE_HMC_STATIC_DIAG_E_ADAPT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/interface_callbacks/interrupt/base_interrupt.hpp>
#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/check_timing.hpp>
#include <stan/services/sample/run_adaptive_sampler.hpp>
#include <stan/mcmc/hmc/static/adapt_diag_e_static_hmc.hpp>
#include <ctime>

namespace stan {
  namespace services {
    namespace sample {

      /**
       * Runs HMC for static integration time with diagonal Euclidean
       * metric with adaptation.
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
       * @param int_time Total integration time for Hamiltonian evolution
       * @param delta Adaptation target acceptance
       * @param kappa Adaptation relaxation exponent
       * @param gamma Adaptation regularization scale
       * @param t0 Adaptation iteration offset
       * @param init_buffer Width of initial fast adaptation interval
       * @param term_buffer Width of final fast adaptation interval
       * @param window Initial width of slow adaptation interval
       * @param interrupt Callback for interrupts
       * @param sample_writer Writer for draws
       * @param diagnostic_writer Writer for diagnostic information
       * @param message_writer Writer for messages
       * @return error code; 0 if no error
       */
      template <class Model, class rng_t>
      int hmc_static_diag_e_adapt(Model& model,
                                  rng_t& base_rng,
                                  Eigen::VectorXd& cont_params,
                                  int num_warmup,
                                  int num_samples,
                                  int num_thin,
                                  bool save_warmup,
                                  int refresh,
                                  double stepsize,
                                  double stepsize_jitter,
                                  double int_time,
                                  double delta,
                                  double gamma,
                                  double kappa,
                                  double t0,
                                  unsigned int init_buffer,
                                  unsigned int term_buffer,
                                  unsigned int window,
                  interface_callbacks::interrupt::base_interrupt& interrupt,
                  interface_callbacks::writer::base_writer& sample_writer,
                  interface_callbacks::writer::base_writer& diagnostic_writer,
                  interface_callbacks::writer::base_writer& message_writer) {
        stan::services::check_timing(model, cont_params, message_writer);

        mcmc::adapt_diag_e_static_hmc<Model, rng_t> sampler(model, base_rng);
        sampler.set_nominal_stepsize_and_T(stepsize, int_time);
        sampler.set_stepsize_jitter(stepsize_jitter);

        sampler.get_stepsize_adaptation().set_mu(log(10 * stepsize));
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_kappa(kappa);
        sampler.get_stepsize_adaptation().set_t0(t0);

        sampler.set_window_params(num_warmup, init_buffer, term_buffer,
                                  window, message_writer);

        run_adaptive_sampler(sampler, model,
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
