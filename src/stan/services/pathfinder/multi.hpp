#ifndef STAN_SERVICES_PATHFINDER_MULTI_HPP
#define STAN_SERVICES_PATHFINDER_MULTI_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/concurrent_writer.hpp>
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
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_queue.h>
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
  using stan::services::pathfinder::internal::write_times;
  const auto start_pathfinders_time = std::chrono::steady_clock::now();
  std::vector<std::string> param_names;
  param_names.push_back("lp_approx__");
  param_names.push_back("lp__");
  param_names.push_back("path__");
  model.constrained_param_names(param_names, true, true);
  parameter_writer(param_names);
  // All work is done in the parallel_for loop
  if (!psis_resample || !calculate_lp) {
    try {
      std::atomic<int> num_path_successes{0};
      stan::callbacks::concurrent_writer safe_write{parameter_writer};
      tbb::parallel_for(
          tbb::blocked_range<int>(0, num_paths),
          [&](tbb::blocked_range<int> r) {
            for (int iter = r.begin(); iter < r.end(); ++iter) {
              // For no psis, have single write to both single and multi writers
              using multi_writer_t = stan::callbacks::tee_writer<
                  SingleParamWriter,
                  stan::callbacks::concurrent_writer<ParamWriter>>;
              multi_writer_t multi_param_writer(
                  single_path_parameter_writer[iter], safe_write);
              auto pathfinder_ret
                  = stan::services::pathfinder::pathfinder_lbfgs_single<false,
                                                                        true>(
                      model, *(init[iter]), random_seed, stride_id + iter,
                      init_radius, history_size, init_alpha, tol_obj,
                      tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
                      num_iterations, num_elbo_draws, num_draws,
                      save_iterations, refresh, interrupt, logger,
                      init_writers[iter], multi_param_writer,
                      single_path_diagnostic_writer[iter], calculate_lp);
              if (pathfinder_ret != error_codes::OK) {
                logger.error(std::string("Pathfinder iteration: ")
                             + std::to_string(iter) + " failed.");
                return;
              }
              num_path_successes++;
            }
          });
      safe_write.wait();
      if (unlikely(num_path_successes == 0)) {
        logger.error("No pathfinders ran successfully.");
        return error_codes::SOFTWARE;
      } else if (unlikely(num_path_successes < num_paths)) {
        std::string msg = std::string("Only ")
                          + std::to_string(num_path_successes.load())
                          + std::string(" of the ") + std::to_string(num_paths)
                          + std::string(" pathfinders succeeded.");
        logger.warn(msg);
      }
    } catch (const std::exception& e) {
      logger.error(e.what());
      return error_codes::SOFTWARE;
    }
    double pathfinders_delta_time = stan::services::util::duration_diff(
        start_pathfinders_time, std::chrono::steady_clock::now());
    write_times<true, false>(parameter_writer, pathfinders_delta_time, 0);
    // Writes are done in loop, so just return
    return error_codes::OK;
  }
  // Save idx of pathfinder and it's elbo for resampling later
  tbb::concurrent_vector<std::pair<Eigen::Index, internal::elbo_est_t>>
      elbo_estimates;
  elbo_estimates.reserve(num_paths);
  try {
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_paths), [&](tbb::blocked_range<int> r) {
          auto non_writer = [](auto&&... /* x */) {};
          for (int iter = r.begin(); iter < r.end(); ++iter) {
            auto pathfinder_ret
                = stan::services::pathfinder::pathfinder_lbfgs_single<true,
                                                                      true>(
                    model, *(init[iter]), random_seed, stride_id + iter,
                    init_radius, history_size, init_alpha, tol_obj, tol_rel_obj,
                    tol_grad, tol_rel_grad, tol_param, num_iterations,
                    num_elbo_draws, num_draws, save_iterations, refresh,
                    interrupt, logger, init_writers[iter], non_writer,
                    single_path_diagnostic_writer[iter], calculate_lp);
            if (unlikely(std::get<0>(pathfinder_ret) != error_codes::OK)) {
              logger.error(std::string("Pathfinder iteration: ")
                           + std::to_string(iter) + " failed.");
              return;
            }
            elbo_estimates.push_back(
                std::make_pair(iter, std::move(std::get<1>(pathfinder_ret))));
          }
        });
    if (unlikely(elbo_estimates.empty())) {
      logger.error("No pathfinders ran successfully.");
      return error_codes::SOFTWARE;
    } else if (unlikely(elbo_estimates.size() < num_paths)) {
      std::string msg = std::string("Only ")
                        + std::to_string(elbo_estimates.size())
                        + std::string(" of the ") + std::to_string(num_paths)
                        + std::string(" pathfinders succeeded.");
      logger.warn(msg);
    }
  } catch (const std::exception& e) {
    logger.error(e.what());
    return error_codes::SOFTWARE;
  }
  double pathfinders_delta_time = stan::services::util::duration_diff(
      start_pathfinders_time, std::chrono::steady_clock::now());
  const auto start_psis_time = std::chrono::steady_clock::now();
  std::sort(elbo_estimates.begin(), elbo_estimates.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  const auto num_successful_paths = elbo_estimates.size();
  const Eigen::Index num_returned_samples = num_draws * num_successful_paths;
  Eigen::Array<double, Eigen::Dynamic, 1> lp_ratios(num_returned_samples);
  Eigen::Index filling_start_row = 0;
  for (const auto& elbo_est : elbo_estimates) {
    const Eigen::Index individ_num_lp = elbo_est.second.lp_ratio.size();
    lp_ratios.segment(filling_start_row, individ_num_lp)
        = elbo_est.second.lp_ratio;
    filling_start_row += individ_num_lp;
  }
  const auto tail_len = std::min(0.2 * num_returned_samples,
                                 3 * std::sqrt(num_returned_samples));
  Eigen::Array<double, Eigen::Dynamic, 1> weight_vals
      = stan::services::psis::psis_weights(lp_ratios, tail_len, logger);
  stan::rng_t rng = util::create_rng(random_seed, stride_id);
  using discrete_dist_t
      = boost::random::discrete_distribution<Eigen::Index, double>;
  boost::variate_generator<stan::rng_t&, discrete_dist_t> rand_psis_idx(
      rng, discrete_dist_t(boost::iterator_range<double*>(
               weight_vals.data(), weight_vals.data() + weight_vals.size())));
  Eigen::Matrix<Eigen::Index, -1, 1> psis_draw_idxs(num_multi_draws);
  for (size_t i = 0; i <= num_multi_draws - 1; ++i) {
    psis_draw_idxs.coeffRef(i) = rand_psis_idx();
  }
  /**
   * The sort helps two main things
   * 1. Uses the same path more frequently so it stays in memory
   * 2. We can write single and multi path samples in one sweep
   */
  std::sort(psis_draw_idxs.data(),
            psis_draw_idxs.data() + psis_draw_idxs.size());
  const auto uc_param_size = param_names.size() - 3;
  std::vector<std::pair<Eigen::Index, Eigen::Index>> single_path_psis_idxs(
      num_successful_paths, {0, 0});
  Eigen::Index prev_path_num = -1;
  for (Eigen::Index i = 0; i < psis_draw_idxs.size(); ++i) {
    auto draw_val = psis_draw_idxs.coeff(i);
    int path_num = std::floor(draw_val / num_draws);
    if (path_num != prev_path_num) {
      single_path_psis_idxs[path_num].first = i;
      prev_path_num = path_num;
    }
    single_path_psis_idxs[path_num].second = i + 1;
  }
  auto constrain_fun = [](auto&& constrained_draws, auto&& unconstrained_draws,
                          auto&& model, auto&& rng) {
    model.write_array(rng, unconstrained_draws, constrained_draws);
    return constrained_draws;
  };
  // If one is null, then all are null
  if (unlikely(single_path_parameter_writer[0].is_valid())) {
    stan::callbacks::concurrent_writer safe_write{parameter_writer};
    tbb::parallel_for(
        tbb::blocked_range<Eigen::Index>(0, num_successful_paths),
        [&](const tbb::blocked_range<Eigen::Index>& r) {
          Eigen::VectorXd unconstrained_col;
          Eigen::VectorXd approx_samples_constrained_col;
          Eigen::Matrix<double, 1, Eigen::Dynamic> sample_row(
              param_names.size());
          for (Eigen::Index i = r.begin(); i < r.end(); ++i) {
            auto psis_writer_position = single_path_psis_idxs[i].first;
            auto path_num = elbo_estimates[i].first;
            auto&& single_writer = single_path_parameter_writer[path_num];
            single_writer(param_names);
            auto&& elbo_est = elbo_estimates[i].second;
            auto&& lp_draws = elbo_est.lp_mat;
            auto&& new_draws = elbo_est.repeat_draws;
            const Eigen::Index num_samples = new_draws.cols();
            stan::rng_t local_rng = util::create_rng(
                random_seed, stride_id + static_cast<std::size_t>(path_num));
            for (Eigen::Index j = 0; j < num_samples; ++j) {
              unconstrained_col = new_draws.col(j);
              constrain_fun(approx_samples_constrained_col, unconstrained_col,
                            model, local_rng);
              sample_row.head(2) = lp_draws.row(j).matrix();
              sample_row(2) = stride_id + path_num;
              sample_row.tail(uc_param_size) = approx_samples_constrained_col;
              single_writer(sample_row);
              while ((elbo_estimates[i].first * num_samples + j)
                     > psis_draw_idxs.coeff(psis_writer_position)) {
                ++psis_writer_position;
              }
              // while() since there can be multiples of the same idx
              while ((elbo_estimates[i].first * num_samples + j)
                     == psis_draw_idxs.coeff(psis_writer_position)) {
                safe_write(sample_row);
                // Since idxs are sorted, just increment the next position.
                ++psis_writer_position;
              }
            }
            write_times<false, false>(single_writer, pathfinders_delta_time, 0);
          }
        });
    safe_write.wait();
    double psis_delta_time = stan::services::util::duration_diff(
        start_psis_time, std::chrono::steady_clock::now());
    write_times<true, true>(parameter_writer, pathfinders_delta_time,
                            psis_delta_time);
    return error_codes::OK;
  }
  stan::callbacks::concurrent_writer safe_write{parameter_writer};
  tbb::parallel_for(
      tbb::blocked_range<Eigen::Index>(0, num_successful_paths),
      [&](const tbb::blocked_range<Eigen::Index>& r) {
        Eigen::VectorXd unconstrained_col;
        Eigen::VectorXd approx_samples_constrained_col;
        Eigen::Matrix<double, 1, Eigen::Dynamic> sample_row(param_names.size());
        for (Eigen::Index i = r.begin(); i < r.end(); ++i) {
          stan::rng_t rng_local = util::create_rng(
              random_seed,
              stride_id + static_cast<std::size_t>(elbo_estimates[i].first));
          for (Eigen::Index j = single_path_psis_idxs[i].first;
               j < single_path_psis_idxs[i].second; ++j) {
            const Eigen::Index draw_idx = psis_draw_idxs.coeff(j);
            double draw_val = static_cast<double>(draw_idx);
            int path_num = static_cast<int>(std::floor(draw_val / num_draws));
            Eigen::Index path_sample_idx = draw_idx % num_draws;
            auto& elbo_est = elbo_estimates[path_num].second;
            auto& lp_draws = elbo_est.lp_mat;
            auto& new_draws = elbo_est.repeat_draws;
            unconstrained_col = new_draws.col(path_sample_idx);
            constrain_fun(approx_samples_constrained_col, unconstrained_col,
                          model, rng_local);
            sample_row.head(2) = lp_draws.row(path_sample_idx).matrix();
            sample_row(2) = stride_id + elbo_estimates[path_num].first;
            sample_row.tail(uc_param_size) = approx_samples_constrained_col;
            safe_write(sample_row);
            // If we see the same draw idx more than once, just increment j and
            // write again if there are remaining draws
            if (j < (psis_draw_idxs.size() - 1)) {
              while (j < (single_path_psis_idxs[i].second)
                     && draw_idx == psis_draw_idxs.coeff(j + 1)) {
                safe_write(sample_row);
                ++j;
              }
            }
          }
        }
      });
  safe_write.wait();
  double psis_delta_time = stan::services::util::duration_diff(
      start_psis_time, std::chrono::steady_clock::now());
  write_times<true, true>(parameter_writer, pathfinders_delta_time,
                          psis_delta_time);
  return error_codes::OK;
}
}  // namespace pathfinder
}  // namespace services
}  // namespace stan
#endif
