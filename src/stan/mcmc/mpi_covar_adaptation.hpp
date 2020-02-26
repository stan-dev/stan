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
  // using est_t = stan::math::mpi::mpi_covar_estimator;
  using est_t = stan::math::welford_covar_estimator;
  int num_chains_;
  int window_size_;
  int init_draw_counter_;
  int draw_req_counter_;
public:
  std::vector<est_t> estimators;
  std::vector<MPI_Request> reqs;
  std::vector<Eigen::MatrixXd> draws;
  std::vector<size_t> num_draws;

  mpi_covar_adaptation(int n_params, int num_chains, int num_iterations, int window_size)
    : num_chains_(num_chains),
      window_size_(window_size),
      init_draw_counter_(0), draw_req_counter_(0),
      estimators(num_iterations / window_size, est_t(n_params)),
      reqs(window_size),
      draws(window_size, Eigen::MatrixXd(n_params, num_chains)),
      num_draws(num_iterations / window_size, 0)
  {}

    void reset_req() {
      draw_req_counter_ = 0;
      reqs.clear();
      reqs.resize(window_size_);
    }

    virtual void add_sample(const Eigen::VectorXd& q, int curr_win_count) {
      const stan::math::mpi::Communicator& comm =
        stan::math::mpi::Session::inter_chain_comm(num_chains_);
      MPI_Iallgather(q.data(), q.size(), MPI_DOUBLE,
                     draws[draw_req_counter_].data(), q.size(), MPI_DOUBLE,
                     comm.comm(), &reqs[draw_req_counter_]);
      draw_req_counter_++;
      for (int win = 0; win < curr_win_count; ++win) {
        num_draws[win]++;
      }
    }

  virtual void learn_metric(Eigen::MatrixXd& covar, int win, int curr_win_count,
                            const stan::math::mpi::Communicator& comm) {
    learn_covariance(covar, win, curr_win_count);
  }

  void learn_covariance(Eigen::MatrixXd& covar, int win, int curr_win_count) {
    int finished = 0;
    int index;
    int flag = 0;
    while(finished < draw_req_counter_) {
      MPI_Testany(draw_req_counter_, reqs.data(), &index, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        finished++;
        for (int i = 0; i < curr_win_count; ++i) {
          for (int chain = 0; chain < num_chains_; ++chain) {
            estimators[i].add_sample(draws[index].col(chain));
          }
        }
      }
    }
    estimators[win].sample_covariance(covar);
    double n = num_draws[win] * num_chains_;
    covar = (n / (n + 5.0)) * covar
      + 1e-3 * (5.0 / (n + 5.0))
      * Eigen::MatrixXd::Identity(covar.rows(), covar.cols());

    reset_req();
  }

  virtual void restart() {
    // estimator.restart();
  }
#else
  public:
    mpi_covar_adaptation(int n_params, int num_chains, int num_iterations, int window_size)
    {}
#endif
};

}  // namespace mcmc

}  // namespace stan



#endif
