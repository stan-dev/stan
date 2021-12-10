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

#define STAN_DEBUG_MULTI_PATH_PSIS false
#define STAN_DEBUG_MULTI_PATH_SINGLE_PATHFINDER false

namespace stan {
namespace services {
namespace optimize {
template <class Model, typename InitContext, typename InitWriter, typename DiagnosticWriter, typename ParamWriter, typename SingleParamWriter, typename SingleDiagnosticWriter>
inline int pathfinder_lbfgs_multi(
    Model& model, InitContext&& init, unsigned int random_seed,
    unsigned int path, double init_radius, int history_size, double init_alpha,
    double tol_obj, double tol_rel_obj, double tol_grad, double tol_rel_grad,
    double tol_param, int num_iterations, bool save_iterations, int refresh,
    callbacks::interrupt& interrupt, size_t num_elbo_draws, size_t num_draws,
    size_t num_multi_draws, size_t num_threads, size_t num_paths,
    callbacks::logger& logger, InitWriter&& init_writers,
    std::vector<SingleParamWriter>& single_path_parameter_writer, std::vector<SingleDiagnosticWriter>& single_path_diagnostic_writer,
    ParamWriter& parameter_writer, DiagnosticWriter& diagnostic_writer) {
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");

  Eigen::Array<double, -1, 1> lp_ratios(num_draws * num_paths);
  std::vector<std::string> param_names;
  model.constrained_param_names(param_names, true, true);
  param_names.push_back("lp_approx__");
  param_names.push_back("lp__");
  parameter_writer(param_names);
  diagnostic_writer(param_names);
  size_t num_params = param_names.size();
  Eigen::Array<double, -1, -1> samples(num_params, num_paths * num_draws);
  tbb::parallel_for(tbb::blocked_range<int>(0, num_paths),
    [&](tbb::blocked_range<int> r) {
      for (int iter = r.begin(); iter < r.end(); ++iter) {
        // TODO: Make fake writer that receives the samples
        auto pathfinder_ret
            = stan::services::optimize::pathfinder_lbfgs_single<true>(
              model, *(init[iter]), random_seed,
              path + iter, init_radius, history_size, init_alpha,
              tol_obj, tol_rel_obj, tol_grad, tol_rel_grad,
              tol_param, num_iterations, save_iterations, refresh,
               interrupt, num_elbo_draws, num_draws,
              num_threads, logger,
              init_writers[iter], single_path_parameter_writer[iter],
              single_path_diagnostic_writer[iter]);
        Eigen::Array<double, -1, 1> lp_ratio = std::get<1>(pathfinder_ret);
        // logic for writing to lp_ratios and draws
        lp_ratios.segment(iter * num_draws, num_draws) = lp_ratio;
        /*
        Eigen::MatrixXd blah1 = samples.middleCols(iter * num_draws, num_draws);
        std::cout << "\n blah rows:" << blah1.rows() << " cols:" << blah1.cols() << "\n";
        Eigen::MatrixXd blah2 = std::get<2>(pathfinder_ret);
        std::cout << "\n samples rows:" << blah2.rows() << " cols:" << blah2.cols() << "\n";
        */
        samples.middleCols(iter * num_draws, num_draws) = std::get<2>(pathfinder_ret);
        if (STAN_DEBUG_MULTI_PATH_SINGLE_PATHFINDER) {
          auto param_vals = std::get<2>(pathfinder_ret).transpose();
          Eigen::RowVectorXd mean_vals = param_vals.colwise().mean();
          std::cout << "Mean Values: \n"
                    << mean_vals.format(CommaInitFmt) << "\n";
          std::cout << "SD Values: \n" << ((param_vals.rowwise() - mean_vals).array().square().matrix().colwise().sum().array() / (param_vals.rows() - 1)).sqrt() << "\n";

        }

      }

    });
  const auto tail_len = std::min(0.2 * samples.cols(), 3 * std::sqrt(samples.cols()));
  Eigen::Array<double, -1, 1> weight_vals = stan::services::psis::get_psis_weights(lp_ratios, tail_len);
  // Figure out if I can use something in boost and not a std::vector
  std::vector<double> lp_weights(num_paths * num_draws);
  for (size_t i = 0; i < weight_vals.size(); ++i) {
    lp_weights[i] = weight_vals[i];
  }
  if (STAN_DEBUG_MULTI_PATH_PSIS) {
    std::cout << "\n tail_len: " << tail_len << "\n";
    std::cout << "\n lp ratios: \n" << lp_ratios.transpose().eval().format(CommaInitFmt);
    std::cout << "\n weight_vals: \n" << weight_vals.transpose().eval().format(CommaInitFmt);
  }
  boost::ecuyer1988 rng
      = util::create_rng<boost::ecuyer1988>(random_seed, path);

  boost::variate_generator<boost::ecuyer1988&,
                           boost::random::discrete_distribution<Eigen::Index, double>>
      rand_psis_idx(
          rng, boost::random::discrete_distribution<Eigen::Index, double>(lp_weights));
  for (size_t i = 0; i <= num_multi_draws; ++i) {
    parameter_writer(samples.col(rand_psis_idx()));
  }
  return 0;
}
}  // namespace optimize
}  // namespace services
}  // namespace stan
#endif
