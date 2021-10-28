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

/**
 *
 */
struct pathfinder_lbfgs_iter_t {
  // The parameters for an iteration of LBFGS
  Eigen::VectorXd params_;
  // Gradients of parameters from an iteration of LBFGS
  Eigen::VectorXd grads_;
  // Vector of diagonal elements of $alpha$
  Eigen::VectorXd alpha_;
  // KxN matrix ($beta$)
  Eigen::MatrixXd beta_;
  // K sized vector $gamma$
  Eigen::VectorXd gamma_;
};

template <typename BaseRNG>
inline auto bfgs_sample(const pathfinder_lbfgs_iter_t& lbfgs_iter, BaseRNG& rng,
                        size_t num_elbo_draws) {
  auto&& params = lbfgs_iter.params_;
  auto&& grads = lbfgs_iter.grads_;
  auto&& alpha = lbfgs_iter.alpha_;
  auto&& beta = lbfgs_iter.beta_;
  auto&& gamma = lbfgs_iter.gamma_;
  const auto N = alpha.size();
  // determinant of a diagonal matrix is the product of elements of its diagonal
  auto alpha_det = alpha[0];
  for (Eigen::Index i = 1; i < N; ++i) {
    alpha_det *= alpha[i];
  }
  Eigen::HouseholderQR<matrix_t> qr(m.rows(), m.cols());
  qr.compute(m);
  const auto min_size = std::min(m.rows(), m.cols());
  Eigen::MatrixXd R = qr.matrixQR().topLeftCorner(min_size, m.cols());
  for (int i = 0; i < min_size; i++) {
    for (int j = 0; j < i; j++) {
      R.coeffRef(i, j) = 0.0;
    }
    if (R(i, i) < 0) {
      R.row(i) *= -1.0;
    }
  }
  Eigen::MatrixXd Q
      = qr.householderQ() * matrix_t::Identity(m.rows(), min_size);
  for (int i = 0; i < min_size; i++) {
    if (qr.matrixQR().coeff(i, i) < 0) {
      Q.col(i) *= -1.0;
    }
  }

  auto L = (Eigen::MatrixXd::Identity(N, N) + R * gamma * R.transpose())
               .llt()
               .matrixL();
  auto log_det_sigma = std::log(alpha_det) + 2 * L.logAbsDeterminant() auto mu
      = params + alpha.asDiagonal() * grads + beta * gamma * beta * grads;
  boost::variate_generator<BaseRNG&, boost::normal_distribution<> >
      rand_unit_gaus(rng, boost::normal_distribution<>());
  Eigen::VectorXd u = Eigen::VectorXd::NullaryExpr(
      N, [&rand_unit_gaus]() { return rand_unit_gaus(); });
  Eigen::MatrixXd approx_draws(N, num_elbo_draws);
  Eigen::MatrixXd log_density_draws(num_elbo_draws);
  for (int m = 0; m < num_elbo_draws; ++m) {
      approx_draws.col(i) = mu + alpha.array().sqrt().asDiagonal() * (Q * L * Q.transpose() * u.col(m) + u.col(m)) - Q * Q.transpose() * u.col(m));
      log_density_draws.coeffRef(i) = -.5 * log_det_sigma
                                      + u.col(m).transpose() * u.col(m)
                                      + N * log(2 * stan::math::pi());
      u = Eigen::VectorXd::NullaryExpr(
          N, [&rand_unit_gaus]() { return rand_unit_gaus(); });
  }
  return std::make_tuple(approx_draws, log_density_draws);
}

/**
 * Runs the L-BFGS algorithm for a model.
 *
 * @tparam Model A model implementation
 * @param[in] model ($log p$ in paper) Input model to test (with data already
 * instantiated)
 * @param[in] init ($\pi_0$ in paper) var context for initialization
 * @param[in] random_seed random seed for the random number generator
 * @param[in] path path id to advance the pseudo random number generator
 * @param[in] init_radius radius to initialize
 * @param[in] history_size  (J in paper) amount of history to keep for L-BFGS
 * @param[in] init_alpha line search step size for first iteration
 * @param[in] tol_obj convergence tolerance on absolute changes in
 *   objective function value
 * @param[in] tol_rel_obj ($\tau^{rel}$ in paper) convergence tolerance on
 * relative changes in objective function value
 * @param[in] tol_grad convergence tolerance on the norm of the gradient
 * @param[in] tol_rel_grad convergence tolerance on the relative norm of
 *   the gradient
 * @param[in] tol_param convergence tolerance on changes in parameter
 *   value
 * @param[in] num_iterations (L in paper) maximum number of iterations
 * @param[in] num_draws_elbo (K in paper) number of MC draws to evaluate ELBO
 * @param[in] num_draws (M in paper) number of approximate posterior draws to
 * return
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
 * 2. Run L-BFGS to return optimization path for parameters, gradients of
 * objective function, and factorization of covariance estimation
 * 3. For each L-BFGS iteration `num_iterations`
 *  3a. Run BFGS-Sample to get `num_elbo_draws` draws from normal approximation
 * and log density of draws in the approximate normal distribution
 *  3b. Calculate a vector of size `num_elbo_draws` joint log probability given
 * normal approximation
 *  3c. Calculate ELBO given 3a and 3b
 * 4. Find $l \in L$ that maximizes ELBO $l^* = arg max_l ELBO^(l)$.
 * 5. Run bfgs-Sample to return `num_draws` draws from ELBO-maximizing normal
 * approx and log density of draws in ELBO-maximizing normal approximation.
 *
 */
template <class Model>
inline int pathfinder_lbfgs_single(
    Model& model, const stan::io::var_context& init, unsigned int random_seed,
    unsigned int path, double init_radius, int history_size, double init_alpha,
    double tol_obj, double tol_rel_obj, double tol_grad, double tol_rel_grad,
    double tol_param, int num_iterations, bool save_iterations, int refresh,
    callbacks::interrupt& interrupt, size_t num_elbo_draws, size_t num_draws,
    size_t num_threads, callbacks::logger& logger,
    callbacks::writer& init_writer, callbacks::writer& parameter_writer,
    callbacks::writer& diagnostic_writer) {
  boost::ecuyer1988 rng = util::create_rng(random_seed, chain);

  std::vector<int> disc_vector;
  // 1. Sample initial parameters
  std::vector<double> cont_vector = util::initialize<false>(
      model, init, rng, init_radius, false, logger, init_writer);
  // Setup LBFGS
  std::stringstream lbfgs_ss;
  using lbfgs_update_t = LBFGSUpdate<double, Eigen::Dynamic>;
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
  Optimizer lbfgs(model, cont_vector, disc_vector, ls_opts, conv_opts,
                  lbfgs_update, &lbfgs_ss);

  std::stringstream initial_msg;
  initial_msg << "Initial log joint probability = " << lbfgs.logp();
  logger.info(initial_msg);

  std::vector<std::string> names;
  names.push_back("lp__");
  model.constrained_param_names(names, true, true);
  parameter_writer(names);
  /*
   * 2. Run L-BFGS to return optimization path for parameters, gradients of
   * objective function, and factorization of covariance estimation
   */
  std::vector<pathfinder_lbfgs_iter_t> lbfgs_iters;
  lbfgs_iters.reserve(max_iter);
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
    // TODO: Need to get out pathfinder_lbfgs_iter_t every step
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

  // 3. For each L-BFGS iteration `num_iterations`
  Eigen::VectorXd lambda(num_iterations);
  for (size_t l = 0; l < num_iterations; l++) {
    /**
     * 3a. Run BFGS-Sample to get `num_elbo_draws` draws from normal
     * approximation and log density of draws in the approximate normal
     * distribution
     */
    auto&& lfbgs_ret_l = lfbgs_ret[l];
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd> phi_and_lq
        = bfgs_sample(lfbgs_ret_l, num_elbo_draws);
    /**
     * 3b. Calculate a vector of size `num_elbo_draws` joint log probability
     * given normal approximation
     */
    Eigen::VectorXd lp_calcs = lp_per_k_calc(std::get<0>(phi_and_lq));
    // 3c. Calculate ELBO given 3a and 3b
    lambda[l] = (lp_calcs - std::get<1>(phi_and_lq)).sum();
  }
  //  4. Find $l \in L$ that maximizes ELBO $l^* = arg max_l ELBO^(l)$.
  Eigen::Index max_l = get_max_idx(lambda);
  auto&& lbfgs_ret_l = lfbgs_ret[max_l];
  /**
   * 5. Run bfgs-Sample to return `num_draws` draws from ELBO-maximizing normal
   * approx and log density of draws in ELBO-maximizing normal approximation.
   */
  std::tuple<Eigen::MatrixXd, Eigen::VectorXd> bfgs_sample(lbfgs_ret_l,
                                                           num_draws);
  return return_code;
}

}  // namespace optimize
}  // namespace services
}  // namespace stan
#endif
