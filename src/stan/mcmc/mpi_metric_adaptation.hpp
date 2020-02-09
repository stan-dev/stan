#ifndef STAN_MCMC_MPI_METRIC_ADAPTATION_HPP
#define STAN_MCMC_MPI_METRIC_ADAPTATION_HPP

#ifdef STAN_LANG_MPI

#include <stan/math/prim/mat.hpp>
#include <vector>

namespace stan {

namespace mcmc {

  class mpi_metric_adaptation {
  public:
    mpi_metric_adaptation() = default;

    virtual void add_sample(const Eigen::VectorXd& q, int curr_win_count) {}

    virtual void learn_metric(Eigen::VectorXd& var, int win, int curr_win_count,
                              const stan::math::mpi::Communicator& comm)
    {}

    virtual void learn_metric(Eigen::MatrixXd& covar, int win, int curr_win_count,
                              const stan::math::mpi::Communicator& comm)
    {}
    virtual void restart() {}
  };

}  // namespace mcmc

}  // namespace stan

#endif

#endif
