#ifndef STAN_MCMC_MPI_VAR_ADAPTATION_HPP
#define STAN_MCMC_MPI_VAR_ADAPTATION_HPP

#include <stan/math/prim/mat.hpp>
#include <stan/math/mpi/mpi_var_estimator.hpp>
#include <vector>

namespace stan {

namespace mcmc {

class mpi_var_adaptation {
 public:
  stan::math::mpi::mpi_var_estimator estimator;

  explicit mpi_var_adaptation(int n_params,
                              const stan::math::mpi::Communicator& comm)
    : estimator(n_params, comm) {}

  explicit mpi_var_adaptation(int num_chains)
    : estimator(0, stan::math::mpi::Session::inter_chain_comm(num_chains)) {}

  void learn_variance(Eigen::VectorXd& var) {
    double n = static_cast<double>(estimator.sample_variance(var));
    var = (n / (n + 5.0)) * var
      + 1e-3 * (5.0 / (n + 5.0)) * Eigen::VectorXd::Ones(var.size());
    estimator.restart();
  }
};

}  // namespace mcmc

}  // namespace stan
#endif
