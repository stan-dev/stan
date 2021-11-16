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
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace stan {
namespace services {
namespace optimize {

template <typename T1, typename T2>
inline Eigen::MatrixXd crossprod(T1&& x, T2&& y) {
  return x.transpose() * y;
}

template <typename T1>
inline Eigen::MatrixXd crossprod(T1&& x) {
  return Eigen::MatrixXd(x.cols(), x.cols())
      .setZero()
      .selfadjointView<Eigen::Lower>()
      .rankUpdate(x.adjoint());
}

template <typename T1, typename T2>
inline Eigen::MatrixXd tcrossprod(T1&& x, T2&& y) {
  return x * y.transpose();
}

template <typename T1>
inline Eigen::MatrixXd tcrossprod(T1&& x) {
  return Eigen::MatrixXd(x.rows(), x.rows())
      .setZero()
      .selfadjointView<Eigen::Lower>()
      .rankUpdate(x);
}
/*

*/

inline bool is_nan(double x) {
  return x == std::numeric_limits<double>::quiet_NaN();
}

inline bool is_infinite(double x) {
  return x == std::numeric_limits<double>::infinity();
}

template <typename EigArray>
inline auto form_init_diag(const EigArray& E0, const EigArray& Yk, const EigArray& Sk) {
  double Dk = (Yk * Sk).sum();
  auto yk_sq = Yk.square().eval();
  double a = ((E0 * yk_sq).sum() / Dk);
  return (a / E0 + yk_sq / Dk
          - a * (Sk / E0).square() / (Sk.square() / E0).sum())
      .inverse()
      .matrix()
      .eval();
}

struct taylor_approx_t {
  Eigen::VectorXd x_center;
  double logdetcholHk;
  Eigen::MatrixXd L_approx;
  Eigen::MatrixXd Qk;
  bool use_full;
  Eigen::MatrixXd wkbart;
  Eigen::MatrixXd mkbart;
  Eigen::VectorXd point_est;
  Eigen::VectorXd grad_est;
  Eigen::MatrixXd ninvRST;
  Eigen::MatrixXd Rkbar;
  Eigen::VectorXd alpha;
};

struct div_est_t {
  double DIV;
  int fn_calls_DIV;
  Eigen::MatrixXd repeat_draws;
  Eigen::VectorXd fn_draws;
  Eigen::VectorXd lp_approx_draws;
};

template <typename Generator>
inline auto calc_u_u2(Generator& rnorm, const taylor_approx_t& taylor_approx, const Eigen::VectorXd& alpha) {
  Eigen::MatrixXd u = rnorm().eval();
  if (taylor_approx.use_full) {
    Eigen::MatrixXd u2
        = crossprod(
              taylor_approx.L_approx, u)
              .colwise()
          + taylor_approx.x_center;
    return std::make_tuple(std::move(u), std::move(u2));
  } else {
    Eigen::MatrixXd u1 = crossprod(taylor_approx.Qk, u);
    Eigen::MatrixXd u2
        = (alpha.array().sqrt().matrix().asDiagonal()
           * (taylor_approx.Qk
                  * crossprod(taylor_approx.L_approx,
                              u1)
              + (u - taylor_approx.Qk * u1)))
              .colwise()
          + taylor_approx.x_center;
    return std::make_tuple(std::move(u), std::move(u2));
  }
}

template <typename SamplePkg, typename F, typename BaseRNG>
auto est_DIV(const SamplePkg& taylor_approx, size_t num_samples, const Eigen::VectorXd& alpha, F&& fn, BaseRNG&& rnorm) {
  const auto D = taylor_approx.x_center.size();
  int draw_ind = 1;
  int fn_calls_DIV = 0;
  auto tuple_u = calc_u_u2(rnorm, taylor_approx, alpha);
  auto&& u = std::get<0>(tuple_u);
  auto&& u2 = std::get<1>(tuple_u);
  // skip bad samples
  Eigen::VectorXd f_test_DIV(u2.cols());
  try {
    for (Eigen::Index i = 0; i < f_test_DIV.size(); ++i) {
      Eigen::VectorXd blahh = u2.col(i).eval();
      f_test_DIV(i) = fn(blahh);
      ++fn_calls_DIV;
    }
  } catch (...) {
    // TODO: Actually catch errors
  }
  Eigen::VectorXd lp_approx_draws
      = -taylor_approx.logdetcholHk - 0.5 * u.array().square().colwise().sum()
        - 0.5 * D * log(2 * stan::math::pi());  // NOTE THIS NEEDS TO BE pi()
  Eigen::MatrixXd repeat_draws = u2;
  //### Divergence estimation ###
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, " ", ", ", "\n", "",
                               "", " ");

  double ELBO = -f_test_DIV.mean() - lp_approx_draws.mean();
  /*
  std::cout << "logdetcholHk: " << taylor_approx.logdetcholHk << "\n";
  std::cout << "ELBO: " << ELBO << "\n";
  std::cout << "repeat_draws: \n" << repeat_draws.transpose().eval().format(CommaInitFmt) << "\n";
  std::cout << "random_stuff: \n" << u.transpose().eval().format(CommaInitFmt) << "\n";
  std::cout << "lp_approx_draws: \n" << lp_approx_draws.format(CommaInitFmt) << "\n";
  std::cout << "fn_call: \n" << f_test_DIV.format(CommaInitFmt) << "\n";
  */
  return div_est_t{ELBO, fn_calls_DIV, std::move(repeat_draws),
                   std::move(f_test_DIV), std::move(lp_approx_draws)};
}

template <typename SamplePkg, typename BaseRNG>
inline auto approximation_samples(const SamplePkg& taylor_approx, size_t num_samples, const Eigen::VectorXd& alpha, BaseRNG&& rnorm) {
  const Eigen::Index num_params = taylor_approx.x_center.size();
  auto tuple_u = calc_u_u2(rnorm, taylor_approx, taylor_approx.alpha);


  auto&& u = std::get<0>(tuple_u);
  auto&& u2 = std::get<1>(tuple_u);
 // TODO: Inline this on the bottom row
  Eigen::VectorXd lp_approx_draws
      = -taylor_approx.logdetcholHk
        - 0.5
              * u.array().square().colwise().sum()
        - 0.5 * num_params * log(2 * stan::math::pi());
  Eigen::MatrixXd final_params(num_params + 1, num_samples);
  final_params.block(0, 0, num_params, num_samples) = std::get<1>(tuple_u);
  final_params.row(num_params) = lp_approx_draws;
  return final_params;
}

template <typename EigVec>
inline auto construct_taylor_approximation_full(
    const Eigen::MatrixXd& Ykt_matt, const Eigen::VectorXd& alpha,
    const Eigen::VectorXd& Dk, const Eigen::MatrixXd& ninvRST,
    const EigVec& point_est, const EigVec& grad_est) {
  auto& Ykt_mat = Ykt_matt.transpose();
  Eigen::MatrixXd y_tcrossprod_alpha = tcrossprod(
      Ykt_mat
      * alpha.head(Ykt_mat.cols()).array().sqrt().matrix().asDiagonal());
  y_tcrossprod_alpha += Dk.asDiagonal();
  Eigen::MatrixXd y_mul_alpha
      = Ykt_mat * alpha.head(Ykt_mat.cols()).asDiagonal();
  Eigen::MatrixXd Hk = crossprod(y_mul_alpha, ninvRST)
                       + crossprod(ninvRST, y_mul_alpha)
                       + crossprod(ninvRST, y_tcrossprod_alpha * ninvRST);
  Hk += alpha.asDiagonal();
  Eigen::MatrixXd cholHk = Hk.llt().matrixL();
  auto logdetcholHk = 2.0 * cholHk.diagonal().array().abs().log().sum();

  Eigen::VectorXd x_center = point_est - Hk * grad_est;
  return taylor_approx_t{std::move(x_center), logdetcholHk, std::move(cholHk),
                         Eigen::MatrixXd(0, 0), true, Eigen::MatrixXd(0, 0), Eigen::MatrixXd(0, 0), point_est, grad_est, ninvRST, alpha};
}

template <typename EigVec>
inline auto construct_taylor_approximation_sparse(
    const Eigen::MatrixXd& Ykt_matt, const Eigen::VectorXd& alpha,
    const Eigen::VectorXd& Dk, const Eigen::MatrixXd& ninvRST,
    const EigVec& point_est, const EigVec& grad_est) {
  auto Ykt_mat = Ykt_matt.transpose().eval();
  const Eigen::Index current_history_size = Ykt_mat.rows();
  Eigen::MatrixXd y_mul_sqrt_alpha
      = Ykt_mat
        * alpha.head(Ykt_mat.cols()).array().sqrt().matrix().asDiagonal();
  Eigen::MatrixXd Wkbart(Ykt_mat.rows() + ninvRST.rows(), Ykt_mat.cols());
  Wkbart.topRows(Ykt_mat.rows()) = y_mul_sqrt_alpha;
  Wkbart.bottomRows(ninvRST.rows())
      = ninvRST * alpha.array().inverse().sqrt().matrix().asDiagonal();

  Eigen::MatrixXd Mkbar(2 * current_history_size, 2 * current_history_size);
  Mkbar.topLeftCorner(current_history_size, current_history_size).setZero();
  Mkbar.topRightCorner(current_history_size, current_history_size) = Eigen::MatrixXd::Identity(current_history_size, current_history_size);
  Mkbar.bottomLeftCorner(current_history_size, current_history_size) = Eigen::MatrixXd::Identity(current_history_size, current_history_size);
  Eigen::MatrixXd y_tcrossprod_alpha = tcrossprod(y_mul_sqrt_alpha);
  y_tcrossprod_alpha += Dk.asDiagonal();
  Mkbar.bottomRightCorner(current_history_size, current_history_size) = y_tcrossprod_alpha;
  Wkbart.transposeInPlace();
  Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(Wkbart);
  const auto min_size = std::min(Wkbart.rows(), Wkbart.cols());
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, " ", ", ", "\n", "",
                               "", " ");
  Eigen::MatrixXd Rkbar = qr.matrixQR().topLeftCorner(min_size, Wkbart.cols());
  Rkbar.triangularView<Eigen::StrictlyLower>().setZero();
  Eigen::MatrixXd Qk
      = qr.householderQ() * Eigen::MatrixXd::Identity(Wkbart.rows (), min_size);
  Eigen::MatrixXd L_approx
      = (Rkbar * Mkbar * Rkbar.transpose()
         + Eigen::MatrixXd::Identity(Rkbar.rows(), Rkbar.rows()))
            .llt()
            .matrixL().transpose();
  double logdetcholHk = L_approx.diagonal().array().abs().log().sum()
                        + 0.5 * alpha.array().log().sum();
  Eigen::VectorXd ninvRSTg = ninvRST * grad_est;
  Eigen::VectorXd alpha_mul_grad = (alpha.array() * grad_est.array()).matrix();
  Eigen::VectorXd x_center_tmp
      = alpha_mul_grad
        + (alpha.array() * crossprod(Ykt_mat, ninvRSTg).array()).matrix()
        + crossprod(ninvRST, Ykt_mat * alpha_mul_grad)
        + crossprod(ninvRST, y_tcrossprod_alpha * ninvRSTg);

  Eigen::VectorXd x_center = point_est - x_center_tmp;
/*
  std::cout << "Full QR: " << qr.matrixQR().format(CommaInitFmt) << "\n";
  std::cout << "Qk: \n" << Qk.format(CommaInitFmt) << "\n";
  std::cout << "L_approx: \n" << L_approx.format(CommaInitFmt) << "\n";
  std::cout << "logdetcholHk: \n" << logdetcholHk << "\n";
  std::cout << "Mkbar: \n" << Mkbar.format(CommaInitFmt) << "\n";
  std::cout << "Wkbar: \n" << Wkbart.format(CommaInitFmt) << "\n";
  std::cout << "x_center: \n" << x_center.format(CommaInitFmt) << "\n";
  std::cout << "NinvRST: " << ninvRST.format(CommaInitFmt) << "\n";
  std::cout << "Ykt_mat: " << Ykt_mat.format(CommaInitFmt) << "\n";
  std::cout << "Rkbar: " << Rkbar.format(CommaInitFmt) << "\n";
*/
  return taylor_approx_t{std::move(x_center), logdetcholHk, std::move(L_approx),
                         std::move(Qk), false, Wkbart, Mkbar, point_est, grad_est, ninvRST, Rkbar, alpha};
}

template <typename EigVec>
inline auto construct_taylor_approximation(const Eigen::MatrixXd& Ykt_mat,
                                           const EigVec& alpha,
                                           const Eigen::VectorXd& Dk,
                                           const Eigen::MatrixXd& ninvRST,
                                           const EigVec& point_est,
                                           const EigVec& grad_est) {
  // If twice the current history size is larger than the number of params
  // use a sparse approximation
  if (2 * Ykt_mat.cols() > Ykt_mat.rows()) {
    return construct_taylor_approximation_full(Ykt_mat, alpha, Dk, ninvRST,
                                               point_est, grad_est);
  } else {
    return construct_taylor_approximation_sparse(Ykt_mat, alpha, Dk, ninvRST,
                                                 point_est, grad_est);
  }
}

template <typename T, stan::require_vector_t<T>* = nullptr>
inline bool check_cond(const T& Yk, const T& Sk) {
  const double Dk = (Yk.array() * Sk.array()).sum();
  if (Dk == 0) {
    return false;
  } else {
    const double thetak = Yk.array().square().sum() / Dk;
    /*
    std::cout << "Yk: \n" << Yk << "\n";
    std::cout << "Sk: \n" << Sk << "\n";
    std::cout << "Dk: " << Dk << "\n";
    std::cout << "Dk: " << thetak << "\n";
    */
    // curvature checking
    if ((Dk <= 0 || std::abs(thetak) > 1e12)) {  // 2.2*e^{-16}
      return false;
    } else {
      return true;
    }
  }
}

template <typename T, stan::require_matrix_t<T>* = nullptr>
inline Eigen::Array<bool, -1, 1> check_cond2(const T& Yk, const T& Sk) {
  auto Dk = (Yk.array() * Sk.array()).colwise().sum().eval();
  auto thetak = (Yk.array().square().colwise().sum() / Dk).abs();
  std::cout << "\nDk: \n" << Dk << "\n";
  std::cout << "\nthetak: \n" << thetak << "\n";
  return (Dk > 0 && thetak <= 1e12).eval();
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
template <typename InputIters, class Model, typename DiagnosticWriter, typename ParamWriter>
inline int pathfinder_lbfgs_single(
    Model& model, InputIters&& input_iters, const stan::io::var_context& init, unsigned int random_seed,
    unsigned int path, double init_radius, int history_size, double init_alpha,
    double tol_obj, double tol_rel_obj, double tol_grad, double tol_rel_grad,
    double tol_param, int num_iterations, bool save_iterations, int refresh,
    callbacks::interrupt& interrupt, size_t num_elbo_draws, size_t num_draws,
    size_t num_threads, callbacks::logger& logger,
    callbacks::writer& init_writer, ParamWriter& parameter_writer,
    DiagnosticWriter& diagnostic_writer) {
  // callbacks::writer& diagnostic_writer) {
  boost::ecuyer1988 rng = util::create_rng(random_seed, path);

  std::vector<int> disc_vector;
  // 1. Sample initial parameters
  std::vector<double> cont_vector = util::initialize<false>(
      model, init, rng, init_radius, false, logger, init_writer);
  const auto param_size = cont_vector.size();

  // Setup LBFGS
  std::stringstream lbfgs_ss;
  stan::optimization::LSOptions<double> ls_opts;
  ls_opts.alpha0 = init_alpha;
  stan::optimization::ConvergenceOptions<double> conv_opts;
  conv_opts.tolAbsF = tol_obj;
  conv_opts.tolRelF = tol_rel_obj;
  conv_opts.tolAbsGrad = tol_grad;
  conv_opts.tolRelGrad = tol_rel_grad;
  conv_opts.tolAbsX = tol_param;
  conv_opts.maxIts = num_iterations;
  using lbfgs_update_t
      = stan::optimization::LBFGSUpdate<double, Eigen::Dynamic>;
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
  diagnostic_writer(names);
  /*
   * 2. Run L-BFGS to return optimization path for parameters, gradients of
   * objective function, and factorization of covariance estimation
   */
  std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> lbfgs_iters;
  int ret = 0;
  Eigen::MatrixXd param_mat(param_size, num_iterations);
  Eigen::MatrixXd grad_mat(param_size, num_iterations);
  {
    std::vector<double> g1;
    double blah = stan::model::log_prob_grad<true, true>(model, cont_vector,
                                                           disc_vector, g1);

    lbfgs_iters.emplace_back(
        Eigen::Map<Eigen::VectorXd>(cont_vector.data(), param_size),
        Eigen::Map<Eigen::VectorXd>(g1.data(), g1.size()));

    param_mat.col(0) = Eigen::Map<Eigen::VectorXd>(cont_vector.data(), param_size);
    grad_mat.col(0) = Eigen::Map<Eigen::VectorXd>(g1.data(), param_size);
  }
  int actual_num_iters = 0;
  while (ret == 0) {
    std::stringstream msg;
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
    double lp = lbfgs.logp();
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
    lbfgs_iters.emplace_back(lbfgs.curr_x(), lbfgs.curr_g());
    ++actual_num_iters;
    param_mat.col(actual_num_iters) = lbfgs.curr_x();
    grad_mat.col(actual_num_iters) = lbfgs.curr_g();
//    model.write_array(rng, cont_vector, disc_vector, values, true, true, &msg);
    if (msg.str().length() > 0) {
      logger.info(msg);
    }
  }
  diagnostic_writer(lbfgs_iters);
  actual_num_iters++;
  /*
  std::cout << "1: \n" << std::get<0>(lbfgs_iters[0]) << "\n";
  std::cout << "2: \n" << std::get<1>(lbfgs_iters[0]) << "\n";
  */
  // 3. For each L-BFGS iteration `num_iterations`
  //Eigen::VectorXd E = Eigen::VectorXd::Ones(param_size);
  Eigen::MatrixXd Ykt_diff = grad_mat.middleCols(1, actual_num_iters) - grad_mat.leftCols(actual_num_iters);
  Eigen::MatrixXd Skt_diff = param_mat.middleCols(1, actual_num_iters) - param_mat.leftCols(actual_num_iters);
  boost::circular_buffer<decltype(Ykt_diff.col(0))> Ykt_h(history_size);
  boost::circular_buffer<decltype(Skt_diff.col(0))> Skt_h(history_size);
  Eigen::Index best_E;// = Eigen::VectorXd::Ones(param_size);
  double div_max = std::numeric_limits<double>::min();
  div_est_t DIV_fit_best;
  taylor_approx_t taylor_approx_best;
  size_t winner = 0;
  Eigen::MatrixXd alpha_mat(param_size, actual_num_iters);
  alpha_mat.col(0).setOnes();
  Eigen::Matrix<bool, -1, 1> check_cond_vec = check_cond2(Ykt_diff, Skt_diff);
  for (Eigen::Index iter = 0; iter < actual_num_iters - 1; iter++) {
    if (check_cond_vec[iter]) {
      alpha_mat.col(iter + 1) = form_init_diag(alpha_mat.col(iter).array(), Ykt_diff.col(iter).array(), Skt_diff.col(iter).array());
    } else {
      alpha_mat.col(iter + 1) = alpha_mat.col(iter);
    }
  }

  boost::variate_generator<boost::ecuyer1988&, boost::normal_distribution<>>
      rand_unit_gaus(rng, boost::normal_distribution<>());
  auto rnorm = [&rand_unit_gaus, num_params = param_size, num_samples = num_draws]() {
    return Eigen::MatrixXd::NullaryExpr(
        num_params, num_samples,
        [&rand_unit_gaus]() { return rand_unit_gaus(); });
  };

  auto fn = [&model](auto&& u) {
    Eigen::VectorXd grad;
    return -stan::model::log_prob_grad<true, true>(model, u, grad, 0);
  };

  // NOTE: We always push the first one no matter what
  check_cond_vec[0] = true;
  for (Eigen::Index iter = 0; iter < actual_num_iters - 1; iter++) {
  //std::cout << "\n------------ Iter: " << iter << "------------\n" << std::endl;
    auto Ykt = Ykt_diff.col(iter);
    auto Skt = Skt_diff.col(iter);
    auto alpha = alpha_mat.col(iter);
    if (check_cond_vec[iter]) {
      // update Y and S matrix
        Ykt_h.push_back(Ykt);
        Skt_h.push_back(Skt);
    }
    /*
    if (Eigen::isnan(alpha.array()).any()) {
      std::cout << "Iter: " << iter << "\n";
      std::cout << "alpha: \n" << alpha << "\n";
      std::cout << "Ykt: \n" << Ykt << "\n";
      std::cout << "Skt: \n" << Skt << "\n";
    } else {
//      std::cout << "alpha: \n" << alpha << "\n" << std::endl;
    }
    */
    const auto current_history_size = Ykt_h.size();
    Eigen::VectorXd Dk(current_history_size);
    for (Eigen::Index i = 0; i < current_history_size; i++) {
      Dk[i] = Ykt_h[i].dot(Skt_h[i]);
    }
    Eigen::MatrixXd Rk = Eigen::MatrixXd::Zero(current_history_size, current_history_size);
    for (Eigen::Index s = 0; s < current_history_size; s++) {
      for (Eigen::Index i = 0; i <= s; i++) {
        Rk(i, s) = Skt_h[i].dot(Ykt_h[s]);
      }
    }
    Eigen::MatrixXd ninvRST;
    {
      Eigen::MatrixXd Skt_mat(param_size, current_history_size);
      for (Eigen::Index i = 0; i < current_history_size; ++i) {
        Skt_mat.col(i) = Skt_h[i];
      }
      Skt_mat.transposeInPlace();
      Rk.triangularView<Eigen::Upper>().solveInPlace(Skt_mat);
      ninvRST = -std::move(Skt_mat);
    }

    Eigen::MatrixXd Ykt_mat(param_size, current_history_size);
    for (Eigen::Index i = 0; i < current_history_size; ++i) {
      Ykt_mat.col(i) = Ykt_h[i];
    }
    /**
     * 3a. Run BFGS-Sample to get `num_elbo_draws` draws from normal
     * approximation and log density of draws in the approximate normal
     * distribution
     */
    taylor_approx_t taylor_appx_tuple = construct_taylor_approximation(
        Ykt_mat, alpha, Dk, ninvRST, param_mat.col(iter),
        grad_mat.col(iter));

    //#lCDIV #lADIV  #lIKL #ELBO
    auto DIV_fit = est_DIV(taylor_appx_tuple, num_elbo_draws, alpha, fn, rnorm);
    // TODO: Calculate total function calls
    // fn_call = fn_call + DIV_fit$fn_calls_DIV
    // DIV_ls = c(DIV_ls, DIV_fit$DIV)
    //  4. Find $l \in L$ that maximizes ELBO $l^* = arg max_l ELBO^(l)$.
    if (DIV_fit.DIV > div_max) {
      div_max = DIV_fit.DIV;
      DIV_fit_best = DIV_fit;
      taylor_approx_best = taylor_appx_tuple;
      winner = iter;
      best_E = iter;
    }
  }
  std::cout << "Winner: " << winner << "\n";
  // Generate upto num_samples samples from the best approximating Normal ##
  /*
  std::cout << "Taylor: \n";
  std::cout << "x_center: \n" << taylor_approx_best.x_center << "\n";
  std::cout << "logdetcholHk: \n" << taylor_approx_best.logdetcholHk << "\n";
  std::cout << "L_approx: \n" << taylor_approx_best.L_approx << "\n";
  std::cout << "Qk: \n" << taylor_approx_best.Qk << "\n";
  */
  auto draws_N_apx
      = approximation_samples(taylor_approx_best, num_draws, alpha_mat.col(best_E), rnorm);

  parameter_writer(draws_N_apx);
  // update the samples in DIV_save ##
  /* Stuff to print
  DIV_save$repeat_draws <- cbind(DIV_save$repeat_draws, draws_N_apx$samples)
  DIV_save$lp_approx_draws <- c(DIV_save$lp_approx_draws,
                                draws_N_apx$lp_apx_draws)
  */

  /* Stuff to print
return(list(taylor_approx_save = taylor_approx_save,
            DIV_save = DIV_save,
            y = y,
            fn_call = fn_call,
            gr_call = gr_call,
            status = "samples"))
            */
  return 1;
}

}  // namespace optimize
}  // namespace services
}  // namespace stan
#endif
