#ifndef STAN_MCMC_COVAR_ADAPTATION_HPP
#define STAN_MCMC_COVAR_ADAPTATION_HPP

#include <stan/math/prim.hpp>
#include <stan/mcmc/windowed_adaptation.hpp>
#include <vector>

namespace stan {

namespace mcmc {

class covar_adaptation : public windowed_adaptation {
 public:
  explicit covar_adaptation(int n)
      : windowed_adaptation("covariance"), estimator_(n) {}

  void learn_covariance(Eigen::MatrixXd& covar, const Eigen::VectorXd& q) {
    if (in_phase2_window())
      estimator_.add_sample(q);

    if (end_phase2_window()) {
      estimator_.sample_covariance(covar);

      double n = static_cast<double>(estimator_.num_samples());
      covar = (n / (n + 5.0)) * covar
              + 1e-3 * (5.0 / (n + 5.0))
                    * Eigen::MatrixXd::Identity(covar.rows(), covar.cols());

      if (!covar.allFinite())
        throw std::runtime_error(
            "Numerical overflow in metric adaptation. "
            "This occurs when the sampler encounters extreme values on the "
            "unconstrained space; this may happen when the posterior density "
            "function is too wide or improper. "
            "There may be problems with your model specification.");
    }
  }

 protected:
  stan::math::welford_covar_estimator estimator_;
};

}  // namespace mcmc

}  // namespace stan

#endif
