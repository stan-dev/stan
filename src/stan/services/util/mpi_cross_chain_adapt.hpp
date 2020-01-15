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

inline double compute_effective_sample_size(std::vector<const double*> draws,
                                            std::vector<size_t> sizes) {
  int num_chains = sizes.size();
  size_t num_draws = sizes[0];
  for (int chain = 1; chain < num_chains; ++chain) {
    num_draws = std::min(num_draws, sizes[chain]);
  }

  // check if chains are constant; all equal to first draw's value
  bool are_all_const = false;
  Eigen::VectorXd init_draw = Eigen::VectorXd::Zero(num_chains);

  for (int chain_idx = 0; chain_idx < num_chains; chain_idx++) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draw(
        draws[chain_idx], sizes[chain_idx]);

    for (int n = 0; n < num_draws; n++) {
      if (!boost::math::isfinite(draw(n))) {
        return std::numeric_limits<double>::quiet_NaN();
      }
    }

    init_draw(chain_idx) = draw(0);

    if (draw.isApproxToConstant(draw(0))) {
      are_all_const |= true;
    }
  }

  if (are_all_const) {
    // If all chains are constant then return NaN
    // if they all equal the same constant value
    if (init_draw.isApproxToConstant(init_draw(0))) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> acov(num_chains);
  Eigen::VectorXd chain_mean(num_chains);
  Eigen::VectorXd chain_var(num_chains);
  for (int chain = 0; chain < num_chains; ++chain) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draw(
        draws[chain], sizes[chain]);
    stan::analyze::autocovariance<double>(draw, acov(chain));
    chain_mean(chain) = draw.mean();
    chain_var(chain) = acov(chain)(0) * num_draws / (num_draws - 1);
  }

  double mean_var = chain_var.mean();
  double var_plus = mean_var * (num_draws - 1) / num_draws;
  if (num_chains > 1)
    var_plus += math::variance(chain_mean);
  Eigen::VectorXd rho_hat_s(num_draws);
  rho_hat_s.setZero();
  Eigen::VectorXd acov_s(num_chains);
  for (int chain = 0; chain < num_chains; ++chain)
    acov_s(chain) = acov(chain)(1);
  double rho_hat_even = 1.0;
  rho_hat_s(0) = rho_hat_even;
  double rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus;
  rho_hat_s(1) = rho_hat_odd;

  // Convert raw autocovariance estimators into Geyer's initial
  // positive sequence. Loop only until num_draws - 4 to
  // leave the last pair of autocorrelations as a bias term that
  // reduces variance in the case of antithetical chains.
  size_t s = 1;
  while (s < (num_draws - 4) && (rho_hat_even + rho_hat_odd) > 0) {
    for (int chain = 0; chain < num_chains; ++chain)
      acov_s(chain) = acov(chain)(s + 1);
    rho_hat_even = 1 - (mean_var - acov_s.mean()) / var_plus;
    for (int chain = 0; chain < num_chains; ++chain)
      acov_s(chain) = acov(chain)(s + 2);
    rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus;
    if ((rho_hat_even + rho_hat_odd) >= 0) {
      rho_hat_s(s + 1) = rho_hat_even;
      rho_hat_s(s + 2) = rho_hat_odd;
    }
    s += 2;
  }

  int max_s = s;
  // this is used in the improved estimate, which reduces variance
  // in antithetic case -- see tau_hat below
  if (rho_hat_even > 0)
    rho_hat_s(max_s + 1) = rho_hat_even;

  // Convert Geyer's initial positive sequence into an initial
  // monotone sequence
  for (int s = 1; s <= max_s - 3; s += 2) {
    if (rho_hat_s(s + 1) + rho_hat_s(s + 2) > rho_hat_s(s - 1) + rho_hat_s(s)) {
      rho_hat_s(s + 1) = (rho_hat_s(s - 1) + rho_hat_s(s)) / 2;
      rho_hat_s(s + 2) = rho_hat_s(s + 1);
    }
  }

  double num_total_draws = num_chains * num_draws;
  // Geyer's truncated estimator for the asymptotic variance
  // Improved estimate reduces variance in antithetic case
  double tau_hat = -1 + 2 * rho_hat_s.head(max_s).sum() + rho_hat_s(max_s + 1);
  return std::min(num_total_draws / tau_hat,
                  num_total_draws * std::log10(num_total_draws));
}
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
inline double compute_effective_sample_size(const double* draw, size_t size) {
  std::vector<const double*> draws{draw};
  std::vector<size_t> sizes{size};
  return compute_effective_sample_size(draws, sizes);
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
                        double chain_stepsize,
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
      chain_gather[nd_win * win + 2] = chain_stepsize;
      chain_gather[nd_win * win + 3] =
        compute_effective_sample_size(draw_p + win * window_size, num_draws);
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
    } else {
      MPI_Gather(chain_gather.data(), n_gather, MPI_DOUBLE,
                 NULL, 0, MPI_DOUBLE, 0, comm.comm());
    }
    MPI_Bcast(res.data(), 1, MPI_DOUBLE, 0, comm.comm());
    return res;
  }
}
}
}
#endif
