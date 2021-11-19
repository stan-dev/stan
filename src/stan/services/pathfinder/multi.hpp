#ifndef STAN_SERVICES_PATHFINDER_SINGLE_HPP
#define STAN_SERVICES_PATHFINDER_SINGLE_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/optimization/bfgs.hpp>
#include <stan/optimization/lbfgs_update.hpp>
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
  Eigen::Array<double, -1, 1> lp_ratios(num_paths * num_draws);
  size_t num_params = 0;
  Eigen::Array<double, -1, 1> samples(num_paths * num_draws, num_params);
  tbb::parallel_for(tbb::blocked_range<int>(0, num_paths),
                    [&](tbb::blocked_range<int> r) {
                      for (int iter = r.begin(); iter < r.end(); ++iter) {
                        auto pathfinder_ret
                            = stan::services::optimize::pathfinder_lbfgs_single(
                                model, empty_context, seed, chain, init_radius,
                                6, 0.001, 1e-12, 10000, 1e-8, 10000000, 1e-8,
                                2000, save_iterations, refresh, callback, 100,
                                100, 1, logger, init, parameter, diagnostics);
                        Eigen::Array<double, -1, 1> lp_ratio
                            = std::get<0>(lp_vals) - std::get<1>(lp_vals);
                        // logic for writing to lp_ratios and draws
                      }
                    });
  // Wonder how I can do relative effective sample sizes?
  Eigen::Array<double, -1, 1> weight_vals = get_psis_weights(lp_ratios, 1);
  // Figure out if I can use something in boost and not a std::vector
  std::vector<double> lp_weights(num_paths * num_draws);
  for (size_t i = 0; i < weight_vals.size(); ++i) {
    lp_weights[i] = weight_vals[i];
  }
  boost::variate_generator<boost::ecuyer1988&,
                           boost::discrete_distribution<Eigen::Index, double>>
      rand_psis_idx(
          rng, boost::discrete_distribution<Eigen::Index, double>(lp_weights));
  for (size_t i = 0; i < num_multi_draws; ++i) {
    param_writer(samples.col(rand_psis_idx()));
  }
  return 1;
}
}  // namespace optimize
}  // namespace services
}  // namespace stan
#endif
