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
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <mutex>

#define STAN_DEBUG_PATH_POST_LBFGS false
#define STAN_DEBUG_PATH_TAYLOR_APPX false
#define STAN_DEBUG_PATH_ELBO_DRAWS false
#define STAN_DEBUG_PATH_CURVE_CHECK false
#define STAN_DEBUG_PATH_BEST_ELBO false
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

template <typename EigVec>
Eigen::MatrixXd circular_buffer_times_diagonal(
    const std::vector<EigVec>& y_buff, const Eigen::VectorXd& alpha) {
  Eigen::MatrixXd ret(y_buff.size(), alpha.size());
  for (Eigen::Index i = 0; i < y_buff.size(); ++i) {
    ret.row(i) = y_buff[i].array() * alpha.array();
  }
  return ret;
}

template <typename EigVec>
Eigen::VectorXd circular_buffer_crossprod_vector(
    const std::vector<EigVec>& y_buff, const Eigen::VectorXd& x) {
  Eigen::VectorXd ret(y_buff[0].size());
  ret.setZero();
  for (Eigen::Index i = 0; i < y_buff.size(); ++i) {
    ret.noalias() += y_buff[i] * x[i];
  }
  return ret;
}

template <typename EigVec>
Eigen::MatrixXd circular_buffer_mul_vector(const std::vector<EigVec>& y_buff,
                                           const Eigen::VectorXd& alpha) {
  Eigen::VectorXd ret(y_buff.size());
  for (Eigen::Index i = 0; i < y_buff.size(); ++i) {
    ret(i) = y_buff[i].dot(alpha);
  }
  return ret;
}

inline bool is_nan(double x) {
  return x == std::numeric_limits<double>::quiet_NaN();
}

inline bool is_infinite(double x) {
  return x == std::numeric_limits<double>::infinity();
}

template <typename EigMat, stan::require_matrix_t<EigMat>* = nullptr>
inline Eigen::Array<bool, -1, 1> check_curvatures(const EigMat& Yk,
                                                  const EigMat& Sk) {
  auto Dk = (Yk.array() * Sk.array()).colwise().sum().eval();
  auto thetak = (Yk.array().square().colwise().sum() / Dk).abs().eval();
  if (STAN_DEBUG_PATH_CURVE_CHECK) {
    std::cout << "\n Check Dk: \n" << Dk.transpose() << "\n";
    std::cout << "\n Check thetak: \n" << thetak.transpose() << "\n";
  }
  return (Dk > 0 && thetak <= 1e12);
}
/*
template <typename EigArray1, typename EigArray2>
inline auto form_diag(const EigArray1& E0, const EigArray2& Yk,
                      const EigArray2& Sk) {
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", " ");
  std::cout << "\n Yk: \n" << Yk.format(CommaInitFmt) << "\n";
  std::cout << "\n Sk: \n" << Sk.format(CommaInitFmt) << "\n";
  std::cout << "\n E0: \n" << E0.format(CommaInitFmt) << "\n";
  double Dk = Yk.matrix().dot(Sk.matrix());
  std::cout << "\n Dk: " << Dk << "\n";
  auto yk_sq = Yk.square().eval();
  std::cout << "\n yk_sq: \n" << yk_sq.format(CommaInitFmt) << "\n";
  double a = ((E0 * yk_sq).sum() / Dk);
  std::cout << "\n a: " << a << "\n";
  auto denom = (a / E0 + yk_sq / Dk
                - a * (Sk / E0).square() / (Sk.square() / E0).sum())
                   .eval();
  std::cout << "\n denom: \n" << denom.format(CommaInitFmt) << "\n";
  auto res = denom.inverse().matrix().eval();
  std::cout << "\n res: \n" << res.format(CommaInitFmt) << "\n";

  return res;
}
*/
/**
 * eq 4.9
 * Gilbert, J.C., Lemaréchal, C. Some numerical experiments with variable-storage quasi-Newton algorithms.
 * Mathematical Programming 45, 407–435 (1989). https://doi.org/10.1007/BF01589113
 */
template <typename EigVec1, typename EigVec2, typename EigVec3>
inline auto form_diag(const EigVec1& alpha_init, const EigVec2& Yk,
                      const EigVec3& Sk) {
  double y_alpha_y = (Yk.dot(alpha_init.asDiagonal() * Yk));
  double y_s = Yk.dot(Sk);
  double s_inv_alpha_s = Sk.dot(alpha_init.array().inverse().matrix().asDiagonal() * Sk);
  return y_s
         / (y_alpha_y / alpha_init.array() + Yk.array().square()
            - (y_alpha_y / s_inv_alpha_s) * (Sk.array() / alpha_init.array()).square());
}

struct taylor_approx_t {
  Eigen::VectorXd x_center;
  double logdetcholHk;
  Eigen::MatrixXd L_approx;
  Eigen::MatrixXd Qk;
  bool use_full;
};

struct div_est_t {
  double DIV;
  int fn_calls_DIV;
  Eigen::MatrixXd repeat_draws;
  Eigen::VectorXd fn_draws;
  Eigen::VectorXd lp_approx_draws;
};

template <typename Generator>
inline auto get_rnorm_and_draws(Generator& rnorm,
                                const taylor_approx_t& taylor_approx,
                                const Eigen::VectorXd& alpha) {
  Eigen::MatrixXd u = rnorm().eval();
  if (taylor_approx.use_full) {
    Eigen::MatrixXd u2 = crossprod(taylor_approx.L_approx, u).colwise()
                         + taylor_approx.x_center;
    return std::make_tuple(std::move(u), std::move(u2));
  } else {
    Eigen::MatrixXd u1 = crossprod(taylor_approx.Qk, u);
    Eigen::MatrixXd u2
        = (alpha.array().sqrt().matrix().asDiagonal()
           * (taylor_approx.Qk * crossprod(taylor_approx.L_approx, u1)
              + (u - taylor_approx.Qk * u1)))
              .colwise()
          + taylor_approx.x_center;
    return std::make_tuple(std::move(u), std::move(u2));
  }
}

template <typename SamplePkg, typename F, typename BaseRNG>
auto est_elbo_draws(const SamplePkg& taylor_approx, size_t num_samples,
                    const Eigen::VectorXd& alpha, F&& fn, BaseRNG&& rnorm) {
  const auto param_size = taylor_approx.x_center.size();
  int draw_ind = 1;
  int fn_calls_DIV = 0;
  auto tuple_u = get_rnorm_and_draws(rnorm, taylor_approx, alpha);
  auto&& u = std::get<0>(tuple_u);
  auto&& u2 = std::get<1>(tuple_u);
  // skip bad samples
  Eigen::VectorXd f_test_elbo_draws(u2.cols());
  try {
    for (Eigen::Index i = 0; i < f_test_elbo_draws.size(); ++i) {
      Eigen::VectorXd blahh = u2.col(i).eval();
      f_test_elbo_draws(i) = fn(blahh);
      ++fn_calls_DIV;
    }
  } catch (...) {
    // TODO: Actually catch errors
  }
  //  std::cout << "\nf_test_elbo_draws: \n" << f_test_elbo_draws << "\n";
  Eigen::VectorXd lp_approx_draws
      = -taylor_approx.logdetcholHk - 0.5 * u.array().square().colwise().sum()
        - 0.5 * param_size
              * log(2 * stan::math::pi());  // NOTE THIS NEEDS TO BE pi()
  Eigen::MatrixXd repeat_draws = u2;
  //### Divergence estimation ###

  double ELBO = -f_test_elbo_draws.mean() - lp_approx_draws.mean();
  if (STAN_DEBUG_PATH_ELBO_DRAWS) {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, " ", ", ", "\n", "",
                                 "", " ");
    std::cout << "logdetcholHk: " << taylor_approx.logdetcholHk << "\n";
    std::cout << "ELBO: " << ELBO << "\n";
    std::cout << "repeat_draws: \n"
              << repeat_draws.transpose().eval().format(CommaInitFmt) << "\n";
    std::cout << "random_stuff: \n"
              << u.transpose().eval().format(CommaInitFmt) << "\n";
    std::cout << "lp_approx_draws: \n"
              << lp_approx_draws.format(CommaInitFmt) << "\n";
    std::cout << "fn_call: \n"
              << f_test_elbo_draws.format(CommaInitFmt) << "\n";
  }
  return div_est_t{ELBO, std::move(fn_calls_DIV), std::move(repeat_draws),
                   std::move(f_test_elbo_draws), std::move(lp_approx_draws)};
}

template <typename SamplePkg, typename BaseRNG>
inline auto approximation_samples(const SamplePkg& taylor_approx,
                                  size_t num_samples,
                                  const Eigen::VectorXd& alpha,
                                  BaseRNG&& rnorm) {
  const Eigen::Index num_params = taylor_approx.x_center.size();
  auto tuple_u = get_rnorm_and_draws(rnorm, taylor_approx, alpha);

  auto&& u = std::get<0>(tuple_u);
  auto&& u2 = std::get<1>(tuple_u);
  // TODO: Inline this on the bottom row
  Eigen::VectorXd lp_approx_draws
      = -taylor_approx.logdetcholHk - 0.5 * u.array().square().colwise().sum()
        - 0.5 * num_params * log(2 * stan::math::pi());
  return std::make_tuple(std::move(std::get<1>(std::move(tuple_u))),
                         std::move(lp_approx_draws));
}

template <typename EigVec, typename Buff>
inline auto construct_taylor_approximation_full(const Buff& Ykt_mat,
                                                const Eigen::VectorXd& alpha,
                                                const Eigen::VectorXd& Dk,
                                                const Eigen::MatrixXd& ninvRST,
                                                const EigVec& point_est,
                                                const EigVec& grad_est) {
  Eigen::MatrixXd y_tcrossprod_alpha
      = tcrossprod(circular_buffer_times_diagonal(
            Ykt_mat, alpha.array().sqrt().matrix().eval()))
        * ninvRST;
  y_tcrossprod_alpha += Dk.asDiagonal();
  Eigen::MatrixXd y_mul_alpha = circular_buffer_times_diagonal(Ykt_mat, alpha);
  Eigen::MatrixXd Hk = crossprod(y_mul_alpha, ninvRST)
                       + crossprod(ninvRST, y_mul_alpha)
                       + crossprod(ninvRST, y_tcrossprod_alpha);
  Hk += alpha.asDiagonal();
  Eigen::MatrixXd cholHk = Hk.llt().matrixL();
  auto logdetcholHk = 2.0 * cholHk.diagonal().array().log().sum();

  Eigen::VectorXd x_center = point_est - Hk * grad_est;
  if (STAN_DEBUG_PATH_TAYLOR_APPX) {
    std::cout << "---Full---\n";

    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n",
                                 "", "", " ");
    std::cout << "Alpha: \n" << alpha.format(CommaInitFmt) << "\n";
    std::cout << "ninvRST: \n" << ninvRST.format(CommaInitFmt) << "\n";
    std::cout << "Dk: \n" << Dk.format(CommaInitFmt) << "\n";
    std::cout << "Point: \n" << point_est.format(CommaInitFmt) << "\n";
    std::cout << "grad: \n" << grad_est.format(CommaInitFmt) << "\n";
    std::cout << "Hk: " << Hk.format(CommaInitFmt) << "\n";
    std::cout << "L_approx: \n" << cholHk.format(CommaInitFmt) << "\n";
    std::cout << "logdetcholHk: \n" << logdetcholHk << "\n";
    std::cout << "x_center: \n" << x_center.format(CommaInitFmt) << "\n";
  }
  return taylor_approx_t{std::move(x_center), logdetcholHk, std::move(cholHk),
                         Eigen::MatrixXd(0, 0), true};
}

template <typename EigVec, typename Buff>
inline auto construct_taylor_approximation_sparse(
    const Buff& Ykt_mat, const Eigen::VectorXd& alpha,
    const Eigen::VectorXd& Dk, const Eigen::MatrixXd& ninvRST,
    const EigVec& point_est, const EigVec& grad_est) {
  const Eigen::Index current_history_size = Ykt_mat.size();
  Eigen::MatrixXd y_mul_sqrt_alpha = circular_buffer_times_diagonal(
      Ykt_mat, alpha.array().sqrt().matrix().eval());
  Eigen::MatrixXd Wkbart(Ykt_mat.size() + ninvRST.rows(), alpha.size());
  Wkbart.topRows(Ykt_mat.size()) = y_mul_sqrt_alpha;
  Wkbart.bottomRows(ninvRST.rows())
      = ninvRST * alpha.array().inverse().sqrt().matrix().asDiagonal();

  Eigen::MatrixXd Mkbar(2 * current_history_size, 2 * current_history_size);
  Mkbar.topLeftCorner(current_history_size, current_history_size).setZero();
  Mkbar.topRightCorner(current_history_size, current_history_size)
      = Eigen::MatrixXd::Identity(current_history_size, current_history_size);
  Mkbar.bottomLeftCorner(current_history_size, current_history_size)
      = Eigen::MatrixXd::Identity(current_history_size, current_history_size);
  Eigen::MatrixXd y_tcrossprod_alpha = tcrossprod(y_mul_sqrt_alpha);
  y_tcrossprod_alpha += Dk.asDiagonal();
  Mkbar.bottomRightCorner(current_history_size, current_history_size)
      = y_tcrossprod_alpha;
  Wkbart.transposeInPlace();
  Eigen::HouseholderQR<Eigen::Ref<decltype(Wkbart)>> qr(Wkbart);
  const auto min_size = std::min(Wkbart.rows(), Wkbart.cols());
  Eigen::MatrixXd Rkbar = qr.matrixQR().topLeftCorner(min_size, Wkbart.cols());
  Rkbar.triangularView<Eigen::StrictlyLower>().setZero();
  Eigen::MatrixXd Qk
      = qr.householderQ() * Eigen::MatrixXd::Identity(Wkbart.rows(), min_size);
  Eigen::MatrixXd L_approx
      = (Rkbar * Mkbar * Rkbar.transpose()
         + Eigen::MatrixXd::Identity(Rkbar.rows(), Rkbar.rows()))
            .llt()
            .matrixL();
  double logdetcholHk = L_approx.diagonal().array().log().sum()
                        + 0.5 * alpha.array().log().sum();
  Eigen::VectorXd ninvRSTg = ninvRST * grad_est;
  Eigen::VectorXd alpha_mul_grad = (alpha.array() * grad_est.array()).matrix();
  Eigen::VectorXd x_center_tmp
      = alpha_mul_grad
        + (alpha.array()
           * circular_buffer_crossprod_vector(Ykt_mat, ninvRSTg).array())
              .matrix()
        + crossprod(ninvRST,
                    circular_buffer_mul_vector(Ykt_mat, alpha_mul_grad))
        + crossprod(ninvRST, y_tcrossprod_alpha * ninvRSTg);

  Eigen::VectorXd x_center = point_est - x_center_tmp;

  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", " ");
  if (STAN_DEBUG_PATH_TAYLOR_APPX) {
    std::cout << "---Sparse---\n";
    std::cout << "Full QR: " << qr.matrixQR().format(CommaInitFmt) << "\n";
    std::cout << "Qk: \n" << Qk.format(CommaInitFmt) << "\n";
    std::cout << "L_approx: \n" << L_approx.format(CommaInitFmt) << "\n";
    std::cout << "logdetcholHk: \n" << logdetcholHk << "\n";
    std::cout << "Mkbar: \n" << Mkbar.format(CommaInitFmt) << "\n";
    std::cout << "Wkbar: \n" << Wkbart.format(CommaInitFmt) << "\n";
    std::cout << "x_center: \n" << x_center.format(CommaInitFmt) << "\n";
    std::cout << "NinvRST: " << ninvRST.format(CommaInitFmt) << "\n";
    std::cout << "Rkbar: " << Rkbar.format(CommaInitFmt) << "\n";
  }
  return taylor_approx_t{std::move(x_center), logdetcholHk, std::move(L_approx),
                         std::move(Qk), false};
}

template <typename EigVec, typename Buff>
inline auto construct_taylor_approximation(const Buff& Ykt_mat,
                                           const EigVec& alpha,
                                           const Eigen::VectorXd& Dk,
                                           const Eigen::MatrixXd& ninvRST,
                                           const EigVec& point_est,
                                           const EigVec& grad_est) {
  // If twice the current history size is larger than the number of params
  // use a sparse approximation
  if (2 * Ykt_mat.size() > Ykt_mat[0].size()) {
    return construct_taylor_approximation_full(Ykt_mat, alpha, Dk, ninvRST,
                                               point_est, grad_est);
  } else {
    return construct_taylor_approximation_sparse(Ykt_mat, alpha, Dk, ninvRST,
                                                 point_est, grad_est);
  }
}
/*
template <typename T, stan::require_vector_t<T>* = nullptr>
inline bool check_curvatures(const T& Yk, const T& Sk) {
  const double Dk = (Yk.array() * Sk.array()).sum();
  if (Dk == 0) {
    return false;
  } else {
    const double thetak = Yk.array().square().sum() / Dk;
    // curvature checking
    if ((Dk <= 0 || std::abs(thetak) > 1e12)) {  // 2.2*e^{-16}
      return false;
    } else {
      return true;
    }
  }
}
*/

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
template <class Model, typename DiagnosticWriter, typename ParamWriter>
inline int pathfinder_lbfgs_single(
    Model& model, const stan::io::var_context& init, unsigned int random_seed,
    unsigned int path, double init_radius, int history_size, double init_alpha,
    double tol_obj, double tol_rel_obj, double tol_grad, double tol_rel_grad,
    double tol_param, int num_iterations, bool save_iterations, int refresh,
    callbacks::interrupt& interrupt, size_t num_elbo_draws, size_t num_draws,
    size_t num_threads, callbacks::logger& logger,
    callbacks::writer& init_writer, ParamWriter& parameter_writer,
    DiagnosticWriter& diagnostic_writer) {
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", " ");

  // callbacks::writer& diagnostic_writer) {
  boost::ecuyer1988 rng
      = util::create_rng<boost::ecuyer1988>(random_seed, path);

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
  Optimizer lbfgs(model, cont_vector, disc_vector, std::move(ls_opts),
                  std::move(conv_opts), std::move(lbfgs_update), &lbfgs_ss);

  std::stringstream initial_msg;
  initial_msg << "Initial log joint probability = " << lbfgs.logp();
  logger.info(initial_msg);

  std::vector<std::string> names;
  model.constrained_param_names(names, true, true);
  names.push_back("lp__");
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

    param_mat.col(0)
        = Eigen::Map<Eigen::VectorXd>(cont_vector.data(), param_size);
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
    //    model.write_array(rng, cont_vector, disc_vector, values, true, true,
    //    &msg);
    if (msg.str().length() > 0) {
      logger.info(msg);
    }
  }
  // diagnostic_writer(lbfgs_iters);
  actual_num_iters;
  /*
  std::cout << "1: \n" << std::get<0>(lbfgs_iters[0]) << "\n";
  std::cout << "2: \n" << std::get<1>(lbfgs_iters[0]) << "\n";
  */
  // 3. For each L-BFGS iteration `num_iterations`
  // Eigen::VectorXd E = Eigen::VectorXd::Ones(param_size);
  Eigen::MatrixXd Ykt_diff = grad_mat.middleCols(1, actual_num_iters)
                             - grad_mat.leftCols(actual_num_iters);
  Eigen::MatrixXd Skt_diff = param_mat.middleCols(1, actual_num_iters)
                             - param_mat.leftCols(actual_num_iters);
  //  boost::circular_buffer<decltype(Ykt_diff.col(0))> Ykt_h(history_size);
  //  boost::circular_buffer<decltype(Skt_diff.col(0))> Skt_h(history_size);
  Eigen::Index best_E;  // = Eigen::VectorXd::Ones(param_size);
  double div_max = std::numeric_limits<double>::lowest();
  div_est_t DIV_fit_best;
  taylor_approx_t taylor_approx_best;
  size_t winner = 0;
  size_t num_curves_correct = 0;
  Eigen::MatrixXd alpha_mat(param_size, actual_num_iters);
  Eigen::Matrix<bool, -1, 1> check_curvatures_vec
      = check_curvatures(Ykt_diff, Skt_diff);
  if (check_curvatures_vec[0]) {
    alpha_mat.col(0) = form_diag(Eigen::Matrix<double, -1, 1>::Ones(param_size),
                                 Ykt_diff.col(0), Skt_diff.col(0));

  } else {
    alpha_mat.col(0).setOnes();
  }
  for (Eigen::Index iter = 1; iter < actual_num_iters; iter++) {
    if (STAN_DEBUG_PATH_CURVE_CHECK) {
      std::cout << "\n---Curve " << iter << "----\n";
    }
    if (check_curvatures_vec[iter]) {
      alpha_mat.col(iter) = form_diag(alpha_mat.col(iter - 1),
                                      Ykt_diff.col(iter), Skt_diff.col(iter));
    } else {
      alpha_mat.col(iter) = alpha_mat.col(iter - 1);
    }
  }
  if (STAN_DEBUG_PATH_POST_LBFGS) {
    std::cout << "\n Alpha mat: "
              << alpha_mat.transpose().eval().format(CommaInitFmt) << "\n";
    std::cout << "\n Ykt_diff mat: "
              << Ykt_diff.transpose().eval().format(CommaInitFmt) << "\n";
    std::cout << "\n grad mat: "
              << grad_mat.leftCols(actual_num_iters + 5)
                     .transpose()
                     .eval()
                     .format(CommaInitFmt)
              << "\n";
    std::cout << "\n Skt_diff mat: "
              << Skt_diff.transpose().eval().format(CommaInitFmt) << "\n";
    std::cout << "\n param mat: "
              << param_mat.leftCols(actual_num_iters + 5)
                     .transpose()
                     .eval()
                     .format(CommaInitFmt)
              << "\n";
  }
  boost::variate_generator<boost::ecuyer1988&, boost::normal_distribution<>>
      rand_unit_gaus(rng, boost::normal_distribution<>());
  auto rnorm
      = [&rand_unit_gaus, num_params = param_size, num_samples = num_draws]() {
          return Eigen::MatrixXd::NullaryExpr(
              num_params, num_samples,
              [&rand_unit_gaus]() { return rand_unit_gaus(); });
        };

  auto fn = [&model](auto&& u) {
    return -model.template log_prob<false, true>(u, 0);
  };

  // NOTE: We always push the first one no matter what
  check_curvatures_vec[0] = true;
  std::mutex update_best_mutex;
  //  for (Eigen::Index iter = 0; iter < actual_num_iters - 1; iter++) {
  tbb::parallel_for(
      tbb::blocked_range<int>(0, actual_num_iters - 1),
      [&](tbb::blocked_range<int> r) {
        for (int iter = r.begin(); iter < r.end(); ++iter) {
          // std::cout << "\n------------ Iter: " << iter << "------------\n";
          // std::endl;
          auto Ykt = Ykt_diff.col(iter);
          auto Skt = Skt_diff.col(iter);
          auto alpha = alpha_mat.col(iter);
          std::vector<size_t> ys_cols;
          const size_t curr_hist_size3
              = iter < history_size ? iter + 1 : history_size;
          {
            for (Eigen::Index end_iter = iter; end_iter >= 0; --end_iter) {
              // std::cout << "End iter: " << end_iter << "\n";
              if (check_curvatures_vec[end_iter]) {
                ys_cols.push_back(end_iter);
              }
              if (ys_cols.size() == history_size) {
                break;
              }
            }
            /*
            std::cout << "hist_size: " << history_size << "\n";
            std::cout << "current_hist_size2: " << current_hist_size2 << "\n";
            std::cout << "curr_hist_size3: " << curr_hist_size3 << "\n";
            std::cout << "Ykt_h size: " << Ykt_h.size() << "\n";
            */
          }
          std::vector<decltype(Ykt_diff.col(0))> Ykt_h;
          std::vector<decltype(Skt_diff.col(0))> Skt_h;
          std::for_each(
              ys_cols.rbegin(), ys_cols.rend(),
              [&Ykt_h, &Skt_h, &Ykt_diff, &Skt_diff](const size_t idx) {
                Ykt_h.push_back(Ykt_diff.col(idx));
                Skt_h.push_back(Skt_diff.col(idx));
              });
          // std::reverse(Ykt_h.begin(), Ykt_h.end());
          // std::reverse(Skt_h.begin(), Skt_h.end());
          Eigen::MatrixXd Ykt_acc(Ykt_h[0].size(), Ykt_h.size());
          for (Eigen::Index i = 0; i < Ykt_h.size(); ++i) {
            Ykt_acc.col(i) = Ykt_h[i];
          }
          // std::cout << "Ykt_mat: " << Ykt_acc.format(CommaInitFmt) << "\n";

          /*
          if (check_curvatures_vec[iter]) {
            // update Y and S matrix
            Ykt_h.push_back(Ykt);
            Skt_h.push_back(Skt);
          }
          */
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
          Eigen::MatrixXd Rk = Eigen::MatrixXd::Zero(current_history_size,
                                                     current_history_size);
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
          /*
          Eigen::MatrixXd Ykt_mat(param_size, current_history_size);
          for (Eigen::Index i = 0; i < current_history_size; ++i) {
            Ykt_mat.col(i) = Ykt_h[i];
          }
          */
          /**
           * 3a. Run BFGS-Sample to get `num_elbo_draws` draws from normal
           * approximation and log density of draws in the approximate normal
           * distribution
           */
          taylor_approx_t taylor_appx_tuple = construct_taylor_approximation(
              Ykt_h, alpha, Dk, ninvRST, param_mat.col(iter),
              grad_mat.col(iter));

          //#lCDIV #lADIV  #lIKL #ELBO
          auto DIV_fit = est_elbo_draws(taylor_appx_tuple, num_elbo_draws,
                                        alpha, fn, rnorm);
          // TODO: Calculate total function calls
          // fn_call = fn_call + DIV_fit$fn_calls_DIV
          // DIV_ls = c(DIV_ls, DIV_fit$DIV)
          //  4. Find $l \in L$ that maximizes ELBO $l^* = arg max_l ELBO^(l)$.
          {
            std::lock_guard<std::mutex> guard(update_best_mutex);
            if (STAN_DEBUG_PATH_BEST_ELBO) {
              std::cout << "DIV curr: " << DIV_fit.DIV << "\n";
              std::cout << "DIV best: " << div_max << "\n";
            }
            if (DIV_fit.DIV > div_max) {
              div_max = DIV_fit.DIV;
              DIV_fit_best = std::move(DIV_fit);
              taylor_approx_best = std::move(taylor_appx_tuple);
              winner = iter;
              best_E = iter;
            }
          }
        }
      },
      tbb::simple_partitioner());
  // std::cout << "Winner: " << winner << "\n";
  // Generate upto num_samples samples from the best approximating Normal ##
  /*
  std::cout << "Taylor: \n";
  std::cout << "x_center: \n" << taylor_approx_best.x_center << "\n";
  std::cout << "logdetcholHk: \n" << taylor_approx_best.logdetcholHk << "\n";
  std::cout << "L_approx: \n" << taylor_approx_best.L_approx << "\n";
  std::cout << "Qk: \n" << taylor_approx_best.Qk << "\n";
  */
  auto draws_tuple = approximation_samples(taylor_approx_best, num_draws,
                                           alpha_mat.col(best_E), rnorm);
  auto&& draws_mat = std::get<0>(draws_tuple);
  auto&& lp_vec = std::get<1>(draws_tuple);
  Eigen::VectorXd unconstrained_draws;
  Eigen::VectorXd constrained_draws1;
  Eigen::VectorXd constrained_draws2(names.size());
  for (Eigen::Index i = 0; i < draws_mat.cols(); ++i) {
    unconstrained_draws = draws_mat.col(i);
    model.write_array(rng, unconstrained_draws, constrained_draws1);
    constrained_draws2.head(names.size() - 1) = constrained_draws1;
    constrained_draws2(constrained_draws2.size() - 1) = lp_vec(i);
    parameter_writer(constrained_draws2);
  }
  return 1;
}

}  // namespace optimize
}  // namespace services
}  // namespace stan
#endif
