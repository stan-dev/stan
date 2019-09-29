#ifndef STAN_MCMC_HMC_HAMILTONIANS_SOFTABS_POINT_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_SOFTABS_POINT_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
namespace mcmc {
/**
 * Point in a phase space with a base
 * Riemannian manifold with SoftAbs metric
 */
class softabs_point : public ps_point {
 public:
  explicit softabs_point(int n)
      : ps_point(n),
        hessian(Eigen::MatrixXd::Identity(n, n)),
        eigen_deco(n),
        softabs_lambda(Eigen::VectorXd::Zero(n)),
        softabs_lambda_inv(Eigen::VectorXd::Zero(n)),
        pseudo_j(Eigen::MatrixXd::Identity(n, n)) {}

  // SoftAbs regularization parameter
  double alpha{1.0};

  Eigen::MatrixXd hessian;

  // Eigendecomposition of the Hessian
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_deco;

  // Log determinant of metric
  double log_det_metric{0};

  // SoftAbs transformed eigenvalues of Hessian
  Eigen::VectorXd softabs_lambda;
  Eigen::VectorXd softabs_lambda_inv;

  // Psuedo-Jacobian of the eigenvalues
  Eigen::MatrixXd pseudo_j;

  inline virtual inline void write_metric(stan::callbacks::writer& writer) {
    writer("No free parameters for SoftAbs metric");
  }
};

}  // namespace mcmc
}  // namespace stan

#endif
