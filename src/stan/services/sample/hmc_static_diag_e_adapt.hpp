#ifndef STAN_SERVICES_SAMPLE_HMC_STATIC_DIAG_E_ADAPT_HPP
#define STAN_SERVICES_SAMPLE_HMC_STATIC_DIAG_E_ADAPT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/interface_callbacks/interrupt/base_interrupt.hpp>
#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/mcmc/hmc/static/adapt_diag_e_static_hmc.hpp>
#include <stan/services/util/run_adaptive_sampler.hpp>
#include <stan/services/util/rng.hpp>
#include <stan/services/util/initialize.hpp>
#include <vector>

namespace stan {
  namespace services {
    namespace sample {

      template <class Model>
      int hmc_static_diag_e_adapt(Model& model,
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
                                  double int_time,
                                  double delta,
                                  double gamma,
                                  double kappa,
                                  double t0,
                                  unsigned int init_buffer,
                                  unsigned int term_buffer,
                                  unsigned int window,
                                interface_callbacks::interrupt::base_interrupt&
                                  interrupt,
                                  interface_callbacks::writer::base_writer&
                                  message_writer,
                                  interface_callbacks::writer::base_writer&
                                  error_writer,
                                  interface_callbacks::writer::base_writer&
                                  sample_writer,
                                  interface_callbacks::writer::base_writer&
                                  diagnostic_writer) {
        boost::ecuyer1988 rng = stan::services::util::rng(random_seed, chain);

        std::vector<int> disc_vector;
        std::vector<double> cont_vector
          = stan::services::util::initialize(model, init, rng, init_radius,
                                             message_writer);

        stan::mcmc::adapt_diag_e_static_hmc<Model, boost::ecuyer1988>
          sampler(model, rng);
        sampler.set_nominal_stepsize_and_T(stepsize, int_time);
        sampler.set_stepsize_jitter(stepsize_jitter);

        sampler.get_stepsize_adaptation().set_mu(log(10 * stepsize));
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_kappa(kappa);
        sampler.get_stepsize_adaptation().set_t0(t0);

        sampler.set_window_params(num_warmup, init_buffer, term_buffer,
                                  window, message_writer);

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
