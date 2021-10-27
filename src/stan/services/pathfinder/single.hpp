#ifndef STAN_SERVICES_PATHFINDER_SINGLE_HPP
#define STAN_SERVICES_PATHFINDER_SINGLE_HPP


#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/optimization/bfgs.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/create_rng.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace stan {
namespace services {
namespace optimize {

using lbfgs_ret_t = std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd>;
template <typename LBFGS>
inline std::vector<lbfgs_ret> lbfgs_with_covar_ret(LBFGS& lbfgs, size_t max_iter) {
  std::vector<lbfgs_ret_t> ret_vec;
  ret_vec.reserve(max_iter);
  int ret = 0;

  while (ret == 0) {
    interrupt();
    if (refresh > 0
        && (lbfgs.iter_num() == 0 || ((lbfgs.iter_num() + 1) % refresh == 0)))
      logger.info(
          "    Iter"
          "      log prob"
          "        ||dx||"
          "      ||grad||"
          "       alpha"
          "      alpha0"
          "  # evals"
          "  Notes ");

    ret = lbfgs.step();
    lp = lbfgs.logp();
    lbfgs.params_r(cont_vector);

    if (refresh > 0
        && (ret != 0 || !lbfgs.note().empty() || lbfgs.iter_num() == 0
            || ((lbfgs.iter_num() + 1) % refresh == 0))) {
      std::stringstream msg;
      msg << " " << std::setw(7) << lbfgs.iter_num() << " ";
      msg << " " << std::setw(12) << std::setprecision(6) << lp << " ";
      msg << " " << std::setw(12) << std::setprecision(6)
          << lbfgs.prev_step_size() << " ";
      msg << " " << std::setw(12) << std::setprecision(6)
          << lbfgs.curr_g().norm() << " ";
      msg << " " << std::setw(10) << std::setprecision(4) << lbfgs.alpha()
          << " ";
      msg << " " << std::setw(10) << std::setprecision(4) << lbfgs.alpha0()
          << " ";
      msg << " " << std::setw(7) << lbfgs.grad_evals() << " ";
      msg << " " << lbfgs.note() << " ";
      logger.info(msg);
    }

    if (lbfgs_ss.str().length() > 0) {
      logger.info(lbfgs_ss);
      lbfgs_ss.str("");
    }

    if (save_iterations) {
      std::vector<double> values;
      std::stringstream msg;
      model.write_array(rng, cont_vector, disc_vector, values, true, true,
                        &msg);
      if (msg.str().length() > 0)
        logger.info(msg);

      values.insert(values.begin(), lp);
      parameter_writer(values);
    }
  }
  return ret_vec;
}


/**
 * Runs the L-BFGS algorithm for a model.
 *
 * @tparam Model A model implementation
 * @param[in] model ($log p$ in paper) Input model to test (with data already instantiated)
 * @param[in] init ($\pi_0$ in paper) var context for initialization
 * @param[in] random_seed random seed for the random number generator
 * @param[in] path path id to advance the pseudo random number generator
 * @param[in] init_radius radius to initialize
 * @param[in] history_size  (J in paper) amount of history to keep for L-BFGS
 * @param[in] init_alpha line search step size for first iteration
 * @param[in] tol_obj convergence tolerance on absolute changes in
 *   objective function value
 * @param[in] tol_rel_obj ($\tau^{rel}$ in paper) convergence tolerance on relative changes in objective function value
 * @param[in] tol_grad convergence tolerance on the norm of the gradient
 * @param[in] tol_rel_grad convergence tolerance on the relative norm of
 *   the gradient
 * @param[in] tol_param convergence tolerance on changes in parameter
 *   value
 * @param[in] num_iterations (L in paper) maximum number of iterations
 * @param[in] num_draws_elbo (K in paper) number of MC draws to evaluate ELBO
 * @param[in] num_draws (M in paper) number of approximate posterior draws to return
 * @param[in] save_iterations indicates whether all the iterations should
 *   be saved to the parameter_writer
 * @param[in] refresh how often to write output to logger
 * @param[in,out] interrupt callback to be called every iteration
 * @param[in,out] logger Logger for messages
 * @param[in,out] init_writer Writer callback for unconstrained inits
 * @param[in,out] parameter_writer output for parameter values
 * @return error_codes::OK if successful
 *
 * The Steps for pathfinder are
 * 1. Sample initial parameters
 * 2. Run L-BFGS to return optimization path for parameters, gradients of objective function, and factorization of covariance estimation
 * 3. For each L-BFGS iteration `num_iterations`
 *  3a. Run BFGS-Sample to get `num_elbo_draws` draws from normal approximation and log density of draws in the approximate normal distribution
 *  3b. Calculate a vector of size `num_elbo_draws` joint log probability given normal approximation
 *  3c. Calculate ELBO given 3a and 3b
 * 4. Find $l \in L$ that maximizes ELBO $l^* = arg max_l ELBO^(l)$.
 * 5. Run BFGSD-Sample to return `num_draws` draws from ELBO-maximizing normal approx and log density of draws in ELBO-maximizing normal approximation.
 *
 */
template <class Model>
inline int pathfinder_lbfgs_single(Model& model, const stan::io::var_context& init,
          unsigned int random_seed, unsigned int path, double init_radius,
          int history_size, double init_alpha, double tol_obj,
          double tol_rel_obj, double tol_grad, double tol_rel_grad,
          double tol_param, int num_iterations, bool save_iterations,
          int refresh, callbacks::interrupt& interrupt, size_t num_elbo_draws,
          size_t num_draws, size_t num_threads,
          callbacks::logger& logger, callbacks::writer& init_writer,
          callbacks::writer& parameter_writer,
          callbacks::writer& diagnostic_writer) {
  boost::ecuyer1988 rng = util::create_rng(random_seed, chain);

  std::vector<int> disc_vector;
  // (1)
  std::vector<double> cont_vector = util::initialize<false>(
      model, init, rng, init_radius, false, logger, init_writer);
  std::stringstream lbfgs_ss;
  using lbfgs_update_t = LBFGSUpdate<double, Eigen::Dynamic> ;
  LSOptions<double> ls_opts;
  ls_opts.alpha0 = init_alpha;
  ConvergenceOptions<double> conv_opts;
  conv_opts.tolAbsF = tol_obj;
  conv_opts.tolRelF = tol_rel_obj;
  conv_opts.tolAbsGrad = tol_grad;
  conv_opts.tolRelGrad = tol_rel_grad;
  conv_opts.tolAbsX = tol_param;
  conv_opts.maxIts = num_iterations;
  lbfgs_update_t lbfgs_update(history_size);
  using Optimizer = stan::optimization::BFGSLineSearch<Model, lbfgs_update_t>;
  Optimizer lbfgs(model, cont_vector, disc_vector, ls_opts, conv_opts, lbfgs_update, &lbfgs_ss);

  std::stringstream initial_msg;
  initial_msg << "Initial log joint probability = " << lbfgs.logp();
  logger.info(initial_msg);

  std::vector<std::string> names;
  names.push_back("lp__");
  model.constrained_param_names(names, true, true);
  parameter_writer(names);
  // (2)
  using lbfgs_ret_t = std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd>;
  std::vector<lbfgs_ret> lbfgs_ret = lbfgs_with_covar_ret(lbfgs, history_size);
  // (3)
  Eigen::VectorXd lambda(num_iterations);
  for (size_t l = 0; l < num_iterations; l++) {
    auto&& lfbgs_ret_l = lfbgs_ret[l];
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd> phi_and_lq = bfgs_sample(std::get<0>(lfbgs_ret_l), std::get<2>(lfbgs_ret_l), std::get<3>(lfbgs_ret_l), std::get<4>(lfbgs_ret_l), num_elbo_draws);
    Eigen::VectorXd lp_calcs = lp_per_k_calc(std::get<0>(phi_and_lq));
    lambda[l] = elbo_calc(lp_calcs, std::get<1>(phi_and_lq));
  }
  // (4)
  Eigen::Index max_l = get_max_idx(lambda);
  // (5)
  auto&& lfbgs_ret_l = lfbgs_ret[max_l];
  std::tuple<Eigen::MatrixXd, Eigen::VectorXd> bfgs_sample(std::get<0>(lfbgs_ret_l), std::get<2>(lfbgs_ret_l), std::get<3>(lfbgs_ret_l), std::get<4>(lfbgs_ret_l), num_draws);
  return return_code;
}

}  // namespace optimize
}  // namespace services
}  // namespace stan
#endif
