#ifndef STAN_SERVICES_UTIL_MPI_CROSS_CHAIN_ADAPT_HPP
#define STAN_SERVICES_UTIL_MPI_CROSS_CHAIN_ADAPT_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <stan/math/mpi/envionment.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <string>

namespace stan {
namespace services {
namespace util {
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
  mpi_cross_chain_adapt(const std::vector<Acc>& acc,
                        const std::vector<double>& chain_stepsize,
                        int num_current_window, int max_num_window,
                        int window_size, int num_chains,
                        double target_rhat,
                        std::vector<double>& chain_gather) {
    using boost::accumulators::accumulator_set;
    using boost::accumulators::stats;
    using boost::accumulators::tag::mean;
    using boost::accumulators::tag::variance;

    using stan::math::mpi::Session;
    using stan::math::mpi::Communicator;

    const Communicator& comm = Session::inter_chain_comm(num_chains);

    const int nd_win = 3; // mean, variance, chain_stepsize
    int n_gather = nd_win * num_current_window;
    for (int win = 0; win < num_current_window; ++win) {
      int n_draws = (num_current_window - win) * window_size;
      double unbiased_var_scale = n_draws / (n_draws - 1.0);
      chain_gather[nd_win * win] = boost::accumulators::mean(acc[win]);
      chain_gather[nd_win * win + 1] = boost::accumulators::variance(acc[win]) *
        unbiased_var_scale;
      chain_gather[nd_win * win + 2] = chain_stepsize[win];
    }

    std::vector<double> res;
    double stepsize = -999.0;

    if (comm.rank() == 0) {
      std::vector<double> rhat(num_current_window), ess(num_current_window);
      std::vector<double> all_chain_gather(n_gather * num_chains);
      MPI_Gather(chain_gather.data(), n_gather, MPI_DOUBLE,
                 all_chain_gather.data(), n_gather, MPI_DOUBLE, 0, comm.comm());
      for (int win = 0; win < num_current_window; ++win) {
        accumulator_set<double, stats<variance>> acc_chain_mean;
        accumulator_set<double, stats<mean>> acc_chain_var;
        for (int chain = 0; chain < num_chains; ++chain) {
          acc_chain_mean(all_chain_gather[chain * n_gather + nd_win * win]);
          acc_chain_var(all_chain_gather[chain * n_gather + nd_win * win + 1]);
        }
        int n_draws = (num_current_window - win) * window_size;
        double var_between = n_draws * boost::accumulators::variance(acc_chain_mean)
          * num_chains / (num_chains - 1);
        double var_within = boost::accumulators::mean(acc_chain_var);
        rhat[win] = sqrt((var_between / var_within + n_draws - 1) / n_draws);

        // TODO also calculate ess
        bool is_adapted = (rhat[win]) < target_rhat;
        if (is_adapted) {
          accumulator_set<double, stats<mean>> acc_step;
          for (int chain = 0; chain < num_chains; ++chain) {
            acc_step(all_chain_gather[chain * n_gather + nd_win * win + 2]);
          }
          stepsize = boost::accumulators::mean(acc_step);
          res.push_back(stepsize);
          res.push_back(rhat[win]);
          break;
        }
      }
      MPI_Bcast(&stepsize, 1, MPI_DOUBLE, 0, comm.comm());
    } else {
      MPI_Gather(chain_gather.data(), n_gather, MPI_DOUBLE,
                 NULL, 0, MPI_DOUBLE, 0, comm.comm());
      MPI_Bcast(&stepsize, 1, MPI_DOUBLE, 0, comm.comm());
      res.push_back(stepsize);
    }
    return res;
  }
}
}
}
#endif
