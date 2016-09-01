#ifndef STAN_SERVICES_SAMPLE_INIT_ADAPT_HPP
#define STAN_SERVICES_SAMPLE_INIT_ADAPT_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/singleton_argument.hpp>
#include <Eigen/Dense>
#include <ostream>

namespace stan {
  namespace services {
    namespace sample {

      template<class Sampler>
      bool init_adapt(Sampler* sampler,
                      const double delta,
                      const double gamma,
                      const double kappa,
                      const double t0,
                      const Eigen::VectorXd& cont_params,
                      interface_callbacks::writer::base_writer& info_writer,
                      interface_callbacks::writer::base_writer& error_writer) {
        const double epsilon = sampler->get_nominal_stepsize();

        sampler->get_stepsize_adaptation().set_mu(log(10 * epsilon));
        sampler->get_stepsize_adaptation().set_delta(delta);
        sampler->get_stepsize_adaptation().set_gamma(gamma);
        sampler->get_stepsize_adaptation().set_kappa(kappa);
        sampler->get_stepsize_adaptation().set_t0(t0);

        sampler->engage_adaptation();

        try {
          sampler->z().q = cont_params;
          sampler->init_stepsize(info_writer, error_writer);
        } catch (const std::exception& e) {
          error_writer("Exception initializing step size.");
          error_writer(e.what());
          return false;
        }
        return true;
      }

      template<class Sampler>
      bool init_adapt(stan::mcmc::base_mcmc* sampler,
                      categorical_argument* adapt,
                      const Eigen::VectorXd& cont_params,
                      interface_callbacks::writer::base_writer& info_writer,
                      interface_callbacks::writer::base_writer& error_writer) {
        double delta
          = dynamic_cast<real_argument*>(adapt->arg("delta"))->value();
        double gamma
          = dynamic_cast<real_argument*>(adapt->arg("gamma"))->value();
        double kappa
          = dynamic_cast<real_argument*>(adapt->arg("kappa"))->value();
        double t0
          = dynamic_cast<real_argument*>(adapt->arg("t0"))->value();

        return init_adapt<Sampler>(dynamic_cast<Sampler*>(sampler),
                                   delta, gamma, kappa, t0, cont_params,
                                   info_writer, error_writer);
      }

    }
  }
}

#endif
