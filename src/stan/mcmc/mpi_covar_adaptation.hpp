#ifndef STAN_MCMC_MPI_COVAR_ADAPTATION_HPP
#define STAN_MCMC_MPI_COVAR_ADAPTATION_HPP

#include <stan/math/prim/mat.hpp>
#include <stan/mcmc/mpi_metric_adaptation.hpp>
#include <vector>

#ifdef STAN_LANG_MPI
#include <stan/math/mpi/mpi_covar_estimator.hpp>
#endif

namespace stan {

namespace mcmc {

  class mpi_covar_adaptation : public mpi_metric_adaptation {
#ifdef STAN_LANG_MPI
  using est_t = stan::math::mpi::mpi_covar_estimator;

  int window_size_;
public:
  est_t estimator;

  mpi_covar_adaptation(int n_params, int num_iterations, int window_size)
    : window_size_(window_size),
      estimator(n_params, num_iterations)
  {}

  virtual void add_sample(const Eigen::VectorXd& q, int curr_win_count) {
    estimator.add_sample(q);
  }

  virtual void learn_metric(Eigen::MatrixXd& covar, int win, int curr_win_count,
                    const stan::math::mpi::Communicator& comm) {
    int col_begin = win * window_size_;
    int num_draws = (curr_win_count - win) * window_size_;
    learn_covariance(covar, col_begin, num_draws, comm);
  }

  void learn_covariance(Eigen::MatrixXd& covar,
                        int col_begin, int n_samples,
                        const stan::math::mpi::Communicator& comm) {
    estimator.sample_covariance(covar, col_begin, n_samples, comm);
    double n = static_cast<double>(estimator.num_samples(comm));
    covar = (n / (n + 5.0)) * covar
      + 1e-3 * (5.0 / (n + 5.0))
      * Eigen::MatrixXd::Identity(covar.rows(), covar.cols());
    // restart();
  }

  virtual void restart() {
    estimator.restart();
  }
#else
  public:
  mpi_covar_adaptation(int n_params, int num_iterations, int window_size)
    {}
#endif
};

}  // namespace mcmc

}  // namespace stan



#endif
