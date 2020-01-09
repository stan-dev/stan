#ifndef STAN_SERVICES_UTIL_CAMPFIRE_WARMUP_HPP
#define STAN_SERVICES_UTIL_CAMPFIRE_WARMUP_HPP

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

/**
 * Generates MCMC transitions.
 *
 * @tparam Model model class
 * @tparam RNG random number generator class
 * @param[in,out] sampler MCMC sampler used to generate transitions
 * @param[in] num_iterations number of MCMC transitions
 * @param[in] start starting iteration number used for printing messages
 * @param[in] finish end iteration number used for printing messages
 * @param[in] num_thin when save is true, a draw will be written to the
 *   mcmc_writer every num_thin iterations
 * @param[in] refresh number of iterations to print a message. If
 *   refresh is zero, iteration number messages will not be printed
 * @param[in] save if save is true, the transitions will be written
 *   to the mcmc_writer. If false, transitions will not be written
 * @param[in] warmup indicates whether these transitions are warmup. Used
 *   for printing iteration number messages
 * @param[in,out] mcmc_writer writer to handle mcmc otuput
 * @param[in,out] init_s starts as the initial unconstrained parameter
 *   values. When the function completes, this will have the final
 *   iteration's unconstrained parameter values
 * @param[in] model model
 * @param[in,out] base_rng random number generator
 * @param[in,out] callback interrupt callback called once an iteration
 * @param[in,out] logger logger for messages
 */
template <class Sampler, class Model, class RNG>
void campfire_warmup(Sampler& sampler, int num_chains,
                     int num_iterations,
                     int start, int finish, int num_thin, int refresh,
                     bool save, bool warmup,
                     util::mcmc_writer& mcmc_writer,
                     stan::mcmc::sample& init_s, Model& model,
                     RNG& base_rng, callbacks::interrupt& callback,
                     callbacks::logger& logger) {
  // for prototyping, we have @c max_num_windows fixed
  const int window_size = 100;
  const int max_num_windows = num_iterations / window_size;

  using boost::accumulators::accumulator_set;
  using boost::accumulators::stats;
  using boost::accumulators::tag::mean;
  using boost::accumulators::tag::variance;

  using stan::math::mpi::Session;
  using stan::math::mpi::Communicator;

  std::vector<accumulator_set<double, stats<mean, variance>>> acc_log(max_num_windows);

  bool is_adapted = false;
  const double target_rhat = 1.05;
  const double target_ess = 50.0;

  int m = 0;
  while (m < num_iterations && (!is_adapted)) {
    callback();

    if (refresh > 0
        && (start + m + 1 == finish || m == 0 || (m + 1) % refresh == 0)) {
      int it_print_width = std::ceil(std::log10(static_cast<double>(finish)));
      std::stringstream message;
      message << "Iteration: ";
      message << std::setw(it_print_width) << m + 1 + start << " / " << finish;
      message << " [" << std::setw(3)
              << static_cast<int>((100.0 * (start + m + 1)) / finish) << "%] ";
      message << (warmup ? " (Warmup)" : " (Sampling)");

      logger.info(message);
    }

    init_s = sampler.transition(init_s, logger);

    if (save && ((m % num_thin) == 0)) {
      mcmc_writer.write_sample_params(base_rng, init_s, sampler, model);
      mcmc_writer.write_diagnostic_params(init_s, sampler);
    }

    double stepsize = -999.0;
    bool is_inter_rank = Session::is_in_inter_chain_comm(num_chains);

    if (is_inter_rank && boost::math::isfinite(init_s.log_prob())) {
      int m_win = m / window_size + 1;
      for (int i = 0; i < m_win; ++i) {
        acc_log[i](init_s.log_prob());
      }

      // though @c boost::acc gives population var instead
      // of sample var, the nb. of draws is supposed to be
      // large enough to make it irrelevant. But for
      // between-chain variance we must correct it because
      // the nb. of chains is not large

      if (m >= window_size && (m + 1) % window_size == 0) {
        int n_gather = 3 * m_win; // mean, variance, stepsize
        std::vector<double> chain_gather(n_gather, 0.0);
        for (int i = 0; i < m_win; ++i) {
          chain_gather[3 * i] = boost::accumulators::mean(acc_log[i]);
          chain_gather[3 * i + 1] = boost::accumulators::variance(acc_log[i]);
          chain_gather[3 * i + 2] = sampler.get_nominal_stepsize();
        }

        const Communicator& comm = Session::inter_chain_comm(num_chains);
        if (comm.rank() == 0) {
          std::vector<double> rhat(m_win), ess(m_win);
          std::vector<double> all_chain_gather(n_gather * num_chains);
          MPI_Gather(chain_gather.data(), n_gather, MPI_DOUBLE,
                     all_chain_gather.data(), n_gather, MPI_DOUBLE, 0, comm.comm());
          for (int i = 0; i < m_win; ++i) {
            accumulator_set<double, stats<variance>> acc_chain_mean;
            accumulator_set<double, stats<mean>> acc_chain_var;
            for (int chain = 0; chain < num_chains; ++chain) {
              acc_chain_mean(all_chain_gather[chain * n_gather + 3 * i]);
              acc_chain_var(all_chain_gather[chain * n_gather + 3 * i + 1]);
            }
            int n_draws = (m_win - i) * window_size;
            double var_between = n_draws * boost::accumulators::variance(acc_chain_mean)
              * num_chains / (num_chains - 1);
            double var_within = boost::accumulators::mean(acc_chain_var);
            rhat[i] = sqrt((var_between / var_within + n_draws - 1) / n_draws);

            // TODO also calculate ess
            is_adapted = (rhat[i]) < target_rhat;
            if (is_adapted) {
              accumulator_set<double, stats<mean>> acc_step;
              for (int chain = 0; chain < num_chains; ++chain) {
                acc_step(all_chain_gather[chain * n_gather + 3 * i + 2]);
              }
              stepsize = boost::accumulators::mean(acc_step);
              std::cout << "taki test rhat: " << rhat[i] << "\n";
              break;
            }
          }
          MPI_Bcast(&stepsize, 1, MPI_DOUBLE, 0, comm.comm());
        } else {
          MPI_Gather(chain_gather.data(), n_gather, MPI_DOUBLE,
                     NULL, 0, MPI_DOUBLE, 0, comm.comm());          
          MPI_Bcast(&stepsize, 1, MPI_DOUBLE, 0, comm.comm());
        }
      }
    }

    const Communicator& intra_comm = Session::intra_chain_comm(num_chains);
    MPI_Bcast(&stepsize, 1, MPI_DOUBLE, 0, intra_comm.comm());
    if (stepsize > 0.0) {
      is_adapted = true;
      sampler.set_nominal_stepsize(stepsize);
      break;
    }

    m++;
  }
}

}  // namespace util
}  // namespace services
}  // namespace stan

#endif
