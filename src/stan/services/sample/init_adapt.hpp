#ifndef STAN_SERVICES_SAMPLE_INIT_ADAPT_HPP
#define STAN_SERVICES_SAMPLE_INIT_ADAPT_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {
    namespace sample {
      /**
       * @tparam Sampler MCMC sampler implementation
       * @param sampler MCMC sampler
       * @param delta Dual averaging target
       * @param gamma Dual averaging scale
       * @param kappa Dual averaging shrinkage
       * @param t0 Dual averaging effective starting iteration
       * @param cont_params Continuous state values
       * @param writer Writer callback for displaying error messages
       */
      template<class Sampler>
      bool init_adapt(Sampler& sampler,
                      const double delta,
                      const double gamma,
                      const double kappa,
                      const double t0,
                      const Eigen::VectorXd& cont_params,
                      std::ostream* o,
                      interface_callbacks::writer::base_writer& writer) {
        const double epsilon = sampler->get_nominal_stepsize();

        sampler.get_stepsize_adaptation().set_mu(log(10 * epsilon));
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_kappa(kappa);
        sampler.get_stepsize_adaptation().set_t0(t0);

        sampler.engage_adaptation();

        try {
          sampler->z().q = cont_params;
          sampler->init_stepsize(writer);
        } catch (const std::exception& e) {
          writer("Error initializing step size.");
          writer(e.what());
          return false;
        }

        return true;
      }

      /**
       * @tparam Sampler MCMC sampler implementation
       * @param sampler MCMC sampler
       * @param adapt Adaptation configuration
       * @param cont_params Continuous state values
       * @param writer Writer callback for displaying error messages
       */
      template<class Sampler>
      bool init_adapt(Sampler& sampler,
                      stan::services::categorical_argument* adapt,
                      const Eigen::VectorXd& cont_params,
                      std::ostream* o,
                      interface_callbacks::writer::base_writer& writer) {
        double delta
          = dynamic_cast<real_argument*>(adapt->arg("delta"))->value();
        double gamma
          = dynamic_cast<real_argument*>(adapt->arg("gamma"))->value();
        double kappa
          = dynamic_cast<real_argument*>(adapt->arg("kappa"))->value();
        double t0
          = dynamic_cast<real_argument*>(adapt->arg("t0"))->value();

        Sampler* s = dynamic_cast<Sampler*>(sampler);

        return init_adapt<Sampler>(s, delta, gamma, kappa, t0, cont_params, o,
                                   writer);
      }

    }  // sample
  }  // services
}  // stan

#endif
