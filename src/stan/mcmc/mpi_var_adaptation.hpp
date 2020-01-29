#ifndef STAN_MCMC_MPI_VAR_ADAPTATION_HPP
#define STAN_MCMC_MPI_VAR_ADAPTATION_HPP

#include <stan/math/prim/mat.hpp>
#include <stan/math/mpi/mpi_var_estimator.hpp>
#include <vector>

namespace stan {

namespace mcmc {

class mpi_var_adaptation {
 public:
  std::vector<stan::math::mpi::mpi_var_estimator> estimators;

  mpi_var_adaptation(int n_params, int num_iterations, int window_size)
    : estimators(num_iterations / window_size,
                 stan::math::mpi::mpi_var_estimator(n_params))
  {}

  void learn_variance(Eigen::VectorXd& var, int win,
                      const stan::math::mpi::Communicator& comm) {
    double n = static_cast<double>(estimators[win].sample_variance(var, comm));
    var = (n / (n + 5.0)) * var
      + 1e-3 * (5.0 / (n + 5.0)) * Eigen::VectorXd::Ones(var.size());
    restart();
  }

  void restart() {
    for (auto&& adapt : estimators) {
      adapt.restart();
    }
  }
};

}  // namespace mcmc

}  // namespace stan
#endif
