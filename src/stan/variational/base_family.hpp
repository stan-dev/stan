#ifndef STAN_VARIATIONAL_BASE_FAMILY_HPP
#define STAN_VARIATIONAL_BASE_FAMILY_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/math/prim.hpp>
#include <algorithm>
#include <ostream>

namespace stan {
namespace variational {
class base_family {
 public:
  // Constructors
  base_family() {}

  /**
   * Return the dimensionality of the approximation.
   */
  virtual int dimension() const = 0;

  // Distribution-based operations
  virtual const Eigen::VectorXd& mean() const = 0;
  virtual double entropy() const = 0;

  Eigen::VectorXd transform(const Eigen::VectorXd& eta) const;

  template <class BaseRNG>
  Eigen::VectorXd sample(BaseRNG& rng, Eigen::VectorXd& eta) const;

  double calc_log_g(Eigen::VectorXd& eta) const;

  template <class BaseRNG>
  Eigen::VectorXd sample_log_g(BaseRNG& rng, Eigen::VectorXd& eta,
                               double& log_g) const;

  template <class M, class BaseRNG>
  void calc_grad(base_family& elbo_grad, M& m, Eigen::VectorXd& cont_params,
                 int n_monte_carlo_grad, BaseRNG& rng,
                 callbacks::logger& logger) const;

 protected:
  void write_error_msg_(std::ostream* error_msgs,
                        const std::exception& e) const {
    if (!error_msgs) {
      return;
    }

    *error_msgs << std::endl
                << "Informational Message: The current gradient evaluation "
                << "of the ELBO is ignored because of the following issue:"
                << std::endl
                << e.what() << std::endl
                << "If this warning occurs often then your model may be "
                << "either severely ill-conditioned or misspecified."
                << std::endl;
  }
};

}  // namespace variational
}  // namespace stan
#endif
