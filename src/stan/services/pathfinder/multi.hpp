#ifndef STAN_SERVICES_PATHFINDER_MULTI_HPP
#define STAN_SERVICES_PATHFINDER_MULTI_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/optimization/bfgs.hpp>
#include <stan/optimization/lbfgs_update.hpp>
#include <stan/services/pathfinder/single.hpp>
#include <stan/services/psis/psis.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/create_rng.hpp>
#include <tbb/parallel_for.h>
#include <boost/random/discrete_distribution.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <mutex>

namespace stan {
namespace services {
namespace optimize {
template <class Model, typename DiagnosticWriter, typename ParamWriter>
inline int pathfinder_lbfgs_multi(
    Model& model, const stan::io::var_context& init, unsigned int random_seed,
    unsigned int path, double init_radius, int history_size, double init_alpha,
    double tol_obj, double tol_rel_obj, double tol_grad, double tol_rel_grad,
    double tol_param, int num_iterations, bool save_iterations, int refresh,
    callbacks::interrupt& interrupt, size_t num_elbo_draws, size_t num_draws,
    size_t num_multi_draws, size_t num_threads, size_t num_paths,
    callbacks::logger& logger, callbacks::writer& init_writer,
    ParamWriter& parameter_writer, DiagnosticWriter& diagnostic_writer) {
  Eigen::Array<double, -1, 1> lp_ratios(num_draws * num_paths);
  size_t num_params = 0;
  Eigen::Array<double, -1, -1> samples(num_paths * num_draws, num_params);
  tbb::parallel_for(tbb::blocked_range<int>(0, num_paths),
    [&](tbb::blocked_range<int> r) {
      for (int iter = r.begin(); iter < r.end(); ++iter) {
        auto pathfinder_ret
            = stan::services::optimize::pathfinder_lbfgs_single<true>(
                model, init, random_seed, path + iter, init_radius,
                history_size, init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad,
                tol_param, num_iterations, save_iterations, refresh,
                interrupt, num_elbo_draws, num_draws, num_threads,
                logger, init_writer, parameter_writer, diagnostic_writer);
        Eigen::Array<double, -1, 1> lp_ratio = std::get<0>(pathfinder_ret);
        // logic for writing to lp_ratios and draws
        lp_ratios.segment(iter * num_draws, num_draws) = -lp_ratio;
        samples.middleRows(iter * num_draws, num_draws) = std::get<1>(pathfinder_ret);
      }
    });
  const auto tail_len = std::min(0.2 * samples.rows(), 3 * std::sqrt(samples.rows()));
  Eigen::Array<double, -1, 1> weight_vals = stan::services::psis::get_psis_weights(lp_ratios, tail_len);
  // Figure out if I can use something in boost and not a std::vector
  std::vector<double> lp_weights(num_paths * num_draws);
  for (size_t i = 0; i < weight_vals.size(); ++i) {
    lp_weights[i] = weight_vals[i];
  }
  boost::ecuyer1988 rng
      = util::create_rng<boost::ecuyer1988>(random_seed, path);

  boost::variate_generator<boost::ecuyer1988&,
                           boost::random::discrete_distribution<Eigen::Index, double>>
      rand_psis_idx(
          rng, boost::random::discrete_distribution<Eigen::Index, double>(lp_weights));
  for (size_t i = 0; i < num_multi_draws; ++i) {
    parameter_writer(samples.row(rand_psis_idx()));
  }
  return 1;
}
}  // namespace optimize
}  // namespace services
}  // namespace stan
#endif
