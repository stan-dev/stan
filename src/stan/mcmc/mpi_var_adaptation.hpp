#ifndef STAN_MCMC_MPI_VAR_ADAPTATION_HPP
#define STAN_MCMC_MPI_VAR_ADAPTATION_HPP

#ifdef STAN_LANG_MPI

#include <stan/math/prim/mat.hpp>
#include <stan/mcmc/mpi_metric_adaptation.hpp>
#include <stan/math/mpi/mpi_var_estimator.hpp>
#include <vector>

namespace stan {

namespace mcmc {

  class mpi_var_adaptation : public mpi_metric_adaptation {
  using est_t = stan::math::mpi::mpi_var_estimator;

public:
  std::vector<est_t> estimators;

  mpi_var_adaptation() = default;

  mpi_var_adaptation(int n_params, int max_num_windows)
    : estimators(max_num_windows, est_t(n_params))
  {}

  mpi_var_adaptation(int n_params, int num_iterations, int window_size)
    : mpi_var_adaptation(n_params, num_iterations / window_size)
  {}

  virtual void add_sample(Eigen::VectorXd& q, int curr_win_count) {
    for (int win = 0; win < curr_win_count; ++win) {
      estimators[win].add_sample(q);
    }
  }

  virtual void learn_metric(Eigen::VectorXd& var, int win, int curr_win_count,
                    const stan::math::mpi::Communicator& comm) {
    learn_variance(var, win, curr_win_count, comm);
  }

  void learn_variance(Eigen::VectorXd& var, int win, int curr_win_count,
                      const stan::math::mpi::Communicator& comm) {
    double n = static_cast<double>(estimators[win].sample_variance(var, comm));
    var = (n / (n + 5.0)) * var
      + 1e-3 * (5.0 / (n + 5.0)) * Eigen::VectorXd::Ones(var.size());
    restart();
  }

  virtual void restart() {
    for (auto&& e : estimators) {
      e.restart();
    }
  }

  // void restart(int n_params, int num_iterations, int window_size) {
  //   estimators.resize(num_iterations / window_size);
  //   for (auto&& e : estimators) {
  //     e.restart(n_params);
  //   }
  // }
};

}  // namespace mcmc

}  // namespace stan

#endif

#endif
