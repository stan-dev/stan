#ifndef STAN__COMMON__INIT_ADAPT_HPP
#define STAN__COMMON__INIT_ADAPT_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/gm/arguments/categorical_argument.hpp>
#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  namespace common {
    
    template<class Sampler>
    bool init_adapt(Sampler* sampler, 
                    const double delta,
                    const double gamma,
                    const double kappa, 
                    const double t0,
                    const Eigen::VectorXd& cont_params) {
      const double epsilon = sampler->get_nominal_stepsize();
      
      sampler->get_stepsize_adaptation().set_mu(log(10 * epsilon));
      sampler->get_stepsize_adaptation().set_delta(delta);
      sampler->get_stepsize_adaptation().set_gamma(gamma);
      sampler->get_stepsize_adaptation().set_kappa(kappa);
      sampler->get_stepsize_adaptation().set_t0(t0);
      
      sampler->engage_adaptation();
      
      try {
        sampler->z().q = cont_params;
        sampler->init_stepsize();
      } catch (const std::exception& e) {
        std::cout << "Exception initializing step size." << std::endl
                  << e.what() << std::endl;
        return false;
      }
      
      return true;
    }

    template<class Sampler>
    bool init_adapt(stan::mcmc::base_mcmc* sampler, 
                    stan::gm::categorical_argument* adapt,
                    const Eigen::VectorXd& cont_params) {
      
      double delta = dynamic_cast<stan::gm::real_argument*>(adapt->arg("delta"))->value();
      double gamma = dynamic_cast<stan::gm::real_argument*>(adapt->arg("gamma"))->value();
      double kappa = dynamic_cast<stan::gm::real_argument*>(adapt->arg("kappa"))->value();
      double t0    = dynamic_cast<stan::gm::real_argument*>(adapt->arg("t0"))->value();
      
      Sampler* s = dynamic_cast<Sampler*>(sampler);

      return init_adapt<Sampler>(s, delta, gamma, kappa, t0, cont_params);
    }

    
  }
}

#endif
