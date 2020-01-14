#ifndef STAN_SERVICES_UTIL_MPI_CROSS_CHAIN_ADAPT_HPP
#define STAN_SERVICES_UTIL_MPI_CROSS_CHAIN_ADAPT_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <stan/math/mpi/envionment.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <stan/analyze/mcmc/split_chains.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <string>

namespace stan {
namespace services {
namespace util {
  /*
   * Computes the effective sample size (ESS) for the specified
   * parameter across all kept samples.  The value returned is the
   * minimum of ESS and the number_total_draws *
   * log10(number_total_draws).
   *
   * This version is based on the one at
   * stan/analyze/mcmc/compute_effective_sample_size.hpp
   * but assuming the chain_mean and chain_var has been
   * calculated(on the fly during adaptation)
   *
   */
inline double
single_chain_ess(const double* draw, size_t num_draws) {
  Eigen::Map<const Eigen::Matrix<double, -1, 1> > d(draw, num_draws);
  Eigen::Matrix<double, -1, 1> acov;
  stan::math::autocorrelation<double>(d, acov);
  double rhos = 0.0;
  int i = 1;
  while (i < num_draws && acov(i) > 0.05) {
    rhos += acov(i);
    i++;
  }
  return double(num_draws) / (1.0 + 2.0 * rhos);
}

  /*
   * @tparam Sampler sampler class
   * @param[in] m_win number of windows
   * @param[in] window_size window size
   * @param[in] num_chains number of chains
   * @param[in,out] chain_gather gathered information from each chain,
   *                must have enough capacity to store up to
   *                maximum windows for all chains.
   # @return vector {stepsize, rhat(only in rank 0)}
   */
  template<typename Acc>
  std::vector<double>
  mpi_cross_chain_adapt(const double* draw_p,
                        const std::vector<Acc>& acc,
                        const std::vector<double>& chain_stepsize,
                        int num_current_window, int max_num_window,
                        int window_size, int num_chains,
                        double target_rhat, double target_ess) {
    using boost::accumulators::accumulator_set;
    using boost::accumulators::stats;
    using boost::accumulators::tag::mean;
    using boost::accumulators::tag::variance;

    using stan::math::mpi::Session;
    using stan::math::mpi::Communicator;


    const Communicator& comm = Session::inter_chain_comm(num_chains);

    const int nd_win = 4; // mean, variance, chain_stepsize
    int n_gather = nd_win * num_current_window;
    std::vector<double> chain_gather(n_gather, 0.0);
    for (int win = 0; win < num_current_window; ++win) {
      int num_draws = (num_current_window - win) * window_size;
      double unbiased_var_scale = num_draws / (num_draws - 1.0);
      chain_gather[nd_win * win] = boost::accumulators::mean(acc[win]);
      chain_gather[nd_win * win + 1] = boost::accumulators::variance(acc[win]) *
        unbiased_var_scale;
      chain_gather[nd_win * win + 2] = chain_stepsize[win];
      chain_gather[nd_win * win + 3] =
        single_chain_ess(draw_p + win * window_size, num_draws);
    }

    double stepsize = -999.0;
    std::vector<double> res(1 + max_num_window, stepsize);

    if (comm.rank() == 0) {
      std::vector<double> all_chain_gather(n_gather * num_chains);
      MPI_Gather(chain_gather.data(), n_gather, MPI_DOUBLE,
                 all_chain_gather.data(), n_gather, MPI_DOUBLE, 0, comm.comm());
      for (int win = 0; win < num_current_window; ++win) {
        accumulator_set<double, stats<variance>> acc_chain_mean;
        accumulator_set<double, stats<mean>> acc_chain_var;
        accumulator_set<double, stats<mean>> acc_step;
        Eigen::VectorXd chain_mean(num_chains);
        Eigen::VectorXd chain_var(num_chains);
        Eigen::ArrayXd chain_ess(num_chains);
        for (int chain = 0; chain < num_chains; ++chain) {
          chain_mean(chain) = all_chain_gather[chain * n_gather + nd_win * win];
          acc_chain_mean(chain_mean(chain));
          chain_var(chain) = all_chain_gather[chain * n_gather + nd_win * win + 1];
          acc_chain_var(chain_var(chain));
          acc_step(all_chain_gather[chain * n_gather + nd_win * win + 2]);
          chain_ess(chain) = all_chain_gather[chain * n_gather + nd_win * win + 3];
        }
        size_t num_draws = (num_current_window - win) * window_size;
        double var_between = num_draws * boost::accumulators::variance(acc_chain_mean)
          * num_chains / (num_chains - 1);
        double var_within = boost::accumulators::mean(acc_chain_var);
        double rhat = sqrt((var_between / var_within + num_draws - 1) / num_draws);
        res[win + 1] = rhat;
        bool is_adapted = rhat < target_rhat && (chain_ess > target_ess).all();
        if (is_adapted) {
          stepsize = boost::accumulators::mean(acc_step);
          res[0] = stepsize;
          break;
        }
      }
      MPI_Bcast(res.data(), 1, MPI_DOUBLE, 0, comm.comm());
    } else {
      MPI_Gather(chain_gather.data(), n_gather, MPI_DOUBLE,
                 NULL, 0, MPI_DOUBLE, 0, comm.comm());
      MPI_Bcast(res.data(), 1, MPI_DOUBLE, 0, comm.comm());
    }
    return res;
  }
}
}
}
#endif
