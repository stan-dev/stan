#ifndef STAN_SERVICES_PATHFINDER_MULTI_HPP
#define STAN_SERVICES_PATHFINDER_MULTI_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/optimization/bfgs.hpp>
#include <stan/optimization/lbfgs_update.hpp>
#include <stan/services/pathfinder/single.hpp>
#include <stan/services/pathfinder/psis.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/duration_diff.hpp>
#include <stan/services/util/initialize.hpp>
#include <tbb/parallel_for.h>
#include <boost/random/discrete_distribution.hpp>
#include <string>
#include <vector>

namespace stan {
namespace services {
namespace pathfinder {

/**
 * Runs multiple pathfinders with final approximate samples drawn using PSIS.
 *
 * @tparam Model A model implementation
 * @tparam InitContext Type inheriting from `stan::io::var_context`
 * @tparam InitWriter Type inheriting from `stan::io::writer`
 * @tparam DiagnosticWriter Type inheriting from `stan::callbacks::writer`
 * @tparam ParamWriter Type inheriting from `stan::callbacks::writer`
 * @tparam SingleDiagnosticWriter Type inheriting from
 * `stan::callbacks::structured_writer`
 * @tparam SingleParamWriter Type inheriting from `stan::callbacks::writer`
 * @param[in] model defining target log density and transforms (log $p$ in
 * paper)
 * @param[in] init ($pi_0$ in paper) var context for initialization. Random
 * initial values will be generated for parameters user has not supplied.
 * @param[in] random_seed seed for the random number generator
 * @param[in] stride_id Id to advance the pseudo random number generator
 * @param[in] init_radius A non-negative value to initialize variables uniformly
 * in (-init_radius, init_radius) if not defined in the initialization var
 * context
 * @param[in] history_size  Non-negative value for (J in paper) amount of
 * history to keep for L-BFGS
 * @param[in] init_alpha Non-negative value for line search step size for first
 * iteration
 * @param[in] tol_obj Non-negative value for convergence tolerance on absolute
 * changes in objective function value
 * @param[in] tol_rel_obj ($tau^{rel}$ in paper) Non-negative value for
 * convergence tolerance on relative changes in objective function value
 * @param[in] tol_grad Non-negative value for convergence tolerance on the norm
 * of the gradient
 * @param[in] tol_rel_grad Non-negative value for convergence tolerance on the
 * relative norm of the gradient
 * @param[in] tol_param Non-negative value for convergence tolerance changes in
 * the L1 norm of parameter values
 * @param[in] num_iterations (L in paper) Non-negative value for maximum number
 * of LBFGS iterations
 * @param[in] save_iterations indicates whether all the iterations should
 *   be saved to the parameter_writer
 * @param[in] refresh Output is written to the logger for each iteration modulo
 * the refresh value
 * @param[in] num_elbo_draws (K in paper) number of MC draws to evaluate ELBO
 * @param[in] num_draws (M in paper) number of approximate posterior draws to
 * return
 * @param[in] num_multi_draws The number of draws to return from PSIS sampling
 * @param[in] num_paths The number of single pathfinders to run.
 * @param[in,out] interrupt callback to be called every iteration
 * @param[in,out] logger Logger for messages
 * @param[in,out] init_writers Writer callback for unconstrained inits
 * @param[in,out] single_path_parameter_writer output for parameter values of
 * the individual pathfinder runs.
 * @param[in,out] single_path_diagnostic_writer output for diagnostics values of
 * the individual pathfinder runs.
 * @param[in,out] parameter_writer output for parameter values
 * @param[in,out] diagnostic_writer output for diagnostics values,
 * `error_codes::SOFTWARE` for failures
 * @param[in] calculate_lp Whether single pathfinder should return lp
 * calculations. If `true`, calculates the joint log probability for each
 * sample. If `false`, (`num_draws` - `num_elbo_draws`) of the joint log
 * probability calculations will be `NA` and psis resampling will not be
 * performed.
 * @param[in] psis_resample If `true`, psis resampling is performed over the
 *  samples returned by all of the individual pathfinders and `num_multi_draws`
 *  samples are written to `parameter_writer`. If `false`, no psis resampling is
 * performed and (`num_paths` * `num_draws`) samples are written to
 * `parameter_writer`.
 * @return error_codes::OK if successful
 */
template <class Model, typename InitContext, typename InitWriter,
          typename DiagnosticWriter, typename ParamWriter,
          typename SingleParamWriter, typename SingleDiagnosticWriter>
inline int pathfinder_lbfgs_multi(
    Model& model, InitContext&& init, unsigned int random_seed,
    unsigned int stride_id, double init_radius, int history_size,
    double init_alpha, double tol_obj, double tol_rel_obj, double tol_grad,
    double tol_rel_grad, double tol_param, int num_iterations,
    int num_elbo_draws, int num_draws, int num_multi_draws, int num_paths,
    bool save_iterations, int refresh, callbacks::interrupt& interrupt,
    callbacks::logger& logger, InitWriter&& init_writers,
    std::vector<SingleParamWriter>& single_path_parameter_writer,
    std::vector<SingleDiagnosticWriter>& single_path_diagnostic_writer,
    ParamWriter& parameter_writer, DiagnosticWriter& diagnostic_writer,
    bool calculate_lp = true, bool psis_resample = true) {
  const auto start_pathfinders_time = std::chrono::steady_clock::now();
  std::vector<std::string> param_names;
  param_names.push_back("lp_approx__");
  param_names.push_back("lp__");
  model.constrained_param_names(param_names, true, true);

  parameter_writer(param_names);
  std::vector<Eigen::Array<double, Eigen::Dynamic, 1>> individual_lp_ratios;
  individual_lp_ratios.resize(num_paths);
  std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>>
      individual_samples;
  individual_samples.resize(num_paths);
  std::atomic<size_t> lp_calls{0};
  try {
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_paths), [&](tbb::blocked_range<int> r) {
          for (int iter = r.begin(); iter < r.end(); ++iter) {
            auto pathfinder_ret
                = stan::services::pathfinder::pathfinder_lbfgs_single<true>(
                    model, *(init[iter]), random_seed, stride_id + iter,
                    init_radius, history_size, init_alpha, tol_obj, tol_rel_obj,
                    tol_grad, tol_rel_grad, tol_param, num_iterations,
                    num_elbo_draws, num_draws, save_iterations, refresh,
                    interrupt, logger, init_writers[iter],
                    single_path_parameter_writer[iter],
                    single_path_diagnostic_writer[iter], calculate_lp);
            if (unlikely(std::get<0>(pathfinder_ret) != error_codes::OK)) {
              logger.error(std::string("Pathfinder iteration: ")
                           + std::to_string(iter) + " failed.");
              return;
            }
            individual_lp_ratios[iter] = std::move(std::get<1>(pathfinder_ret));
            individual_samples[iter] = std::move(std::get<2>(pathfinder_ret));
            lp_calls += std::get<3>(pathfinder_ret);
          }
        });
  } catch (const std::exception& e) {
    logger.error(e.what());
    return error_codes::SOFTWARE;
  }

  // if any pathfinders failed, we want to remove their empty results
  individual_lp_ratios.erase(
      std::remove_if(individual_lp_ratios.begin(), individual_lp_ratios.end(),
                     [](const auto& v) { return v.size() == 0; }),
      individual_lp_ratios.end());
  individual_samples.erase(
      std::remove_if(individual_samples.begin(), individual_samples.end(),
                     [](const auto& v) { return v.size() == 0; }),
      individual_samples.end());

  const auto end_pathfinders_time = std::chrono::steady_clock::now();

  const double pathfinders_delta_time = stan::services::util::duration_diff(
      start_pathfinders_time, end_pathfinders_time);
  const auto start_psis_time = std::chrono::steady_clock::now();
  const size_t successful_pathfinders = individual_samples.size();
  if (successful_pathfinders == 0) {
    logger.info("No pathfinders ran successfully");
    return error_codes::SOFTWARE;
  }
  if (refresh != 0) {
    logger.info("Total log probability function evaluations:"
                + std::to_string(lp_calls));
  }
  size_t num_returned_samples = 0;
  // Because of failure in single pathfinder we can have multiple returned sizes
  for (auto&& ilpr : individual_lp_ratios) {
    num_returned_samples += ilpr.size();
  }
  // Rows are individual parameters and columns are samples per iteration
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> samples(
      individual_samples[0].rows(), num_returned_samples);
  Eigen::Index filling_start_row = 0;
  for (size_t i = 0; i < successful_pathfinders; ++i) {
    const Eigen::Index individ_num_samples = individual_samples[i].cols();
    samples.middleCols(filling_start_row, individ_num_samples)
        = individual_samples[i].matrix();
    filling_start_row += individ_num_samples;
  }
  double psis_delta_time = 0;
  if (psis_resample && calculate_lp) {
    Eigen::Array<double, Eigen::Dynamic, 1> lp_ratios(num_returned_samples);
    filling_start_row = 0;
    for (size_t i = 0; i < successful_pathfinders; ++i) {
      const Eigen::Index individ_num_samples = individual_lp_ratios[i].size();
      lp_ratios.segment(filling_start_row, individ_num_samples)
          = individual_lp_ratios[i];
      filling_start_row += individ_num_samples;
    }

    const auto tail_len = std::min(0.2 * num_returned_samples,
                                   3 * std::sqrt(num_returned_samples));
    Eigen::Array<double, Eigen::Dynamic, 1> weight_vals
        = stan::services::psis::psis_weights(lp_ratios, tail_len, logger);
    stan::rng_t rng = util::create_rng(random_seed, stride_id);
    boost::variate_generator<stan::rng_t&, boost::random::discrete_distribution<
                                               Eigen::Index, double>>
        rand_psis_idx(
            rng, boost::random::discrete_distribution<Eigen::Index, double>(
                     boost::iterator_range<double*>(
                         weight_vals.data(),
                         weight_vals.data() + weight_vals.size())));
    for (size_t i = 0; i <= num_multi_draws - 1; ++i) {
      parameter_writer(samples.col(rand_psis_idx()));
    }
    const auto end_psis_time = std::chrono::steady_clock::now();
    psis_delta_time
        = stan::services::util::duration_diff(start_psis_time, end_psis_time);

  } else {
    parameter_writer(samples);
  }
  parameter_writer();
  const auto time_header = std::string("Elapsed Time: ");
  std::string optim_time_str
      = time_header + std::to_string(pathfinders_delta_time)
        + std::string(" seconds")
        + ((psis_resample && calculate_lp) ? " (Pathfinders)" : " (Total)");
  parameter_writer(optim_time_str);
  if (psis_resample && calculate_lp) {
    std::string psis_time_str = std::string(time_header.size(), ' ')
                                + std::to_string(psis_delta_time)
                                + " seconds (PSIS)";
    parameter_writer(psis_time_str);
    std::string total_time_str
        = std::string(time_header.size(), ' ')
          + std::to_string(pathfinders_delta_time + psis_delta_time)
          + " seconds (Total)";
    parameter_writer(total_time_str);
  }
  parameter_writer();
  return error_codes::OK;
}
}  // namespace pathfinder
}  // namespace services
}  // namespace stan
#endif
