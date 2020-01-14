#ifndef STAN_SERVICES_UTIL_CAMPFIRE_WARMUP_HPP
#define STAN_SERVICES_UTIL_CAMPFIRE_WARMUP_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <stan/services/util/mpi_cross_chain_adapt.hpp>
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
                     int window_size, double target_rhat, double target_ess,
                     util::mcmc_writer& mcmc_writer,
                     stan::mcmc::sample& init_s, Model& model,
                     RNG& base_rng, callbacks::interrupt& callback,
                     callbacks::logger& logger) {
  using boost::accumulators::accumulator_set;
  using boost::accumulators::stats;
  using boost::accumulators::tag::mean;
  using boost::accumulators::tag::variance;

  using stan::math::mpi::Session;
  using stan::math::mpi::Communicator;

  const int max_num_windows = num_iterations / window_size;
  std::vector<accumulator_set<double, stats<mean, variance>>>
    acc_log(max_num_windows);
  std::vector<double> acov(max_num_windows, 0.0);

  bool is_adapted = false;

  int m = 0;
  std::vector<double> draw;
  draw.reserve(num_iterations);
  double stepsize = -999.0;
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

    const Communicator& inter_comm = Session::inter_chain_comm(num_chains);
    bool is_inter_rank = Session::is_in_inter_chain_comm(num_chains);
    int m_win = m / window_size + 1;

    // incrementally add data
    if (is_inter_rank) {
      draw.push_back(init_s.log_prob());
      for (int i = 0; i < m_win; ++i) {
        acc_log[i](init_s.log_prob());
      }
    }

    if (boost::math::isfinite(init_s.log_prob())) {
      const Communicator& intra_comm = Session::intra_chain_comm(num_chains);
      if ((m + 1) % window_size == 0) {
        if (is_inter_rank) {
          std::vector<double> adapt_result =
            stan::services::util::mpi_cross_chain_adapt(draw.data(), acc_log,
                                                        sampler.get_nominal_stepsize(),
                                                        m_win, max_num_windows,
                                                        window_size, num_chains, target_rhat, target_ess);
          stepsize = adapt_result[0];
        }
        MPI_Bcast(&stepsize, 1, MPI_DOUBLE, 0, intra_comm.comm());
        if (stepsize > 0.0) {
          is_adapted = true;
        }
      }
    }
    m++;
  }
  sampler.set_nominal_stepsize(stepsize);
}

}  // namespace util
}  // namespace services
}  // namespace stan

#endif
