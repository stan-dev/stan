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

#define STAN_DEBUG_PATH_ALL false
#define STAN_DEBUG_PATH_POST_LBFGS false || STAN_DEBUG_PATH_ALL
#define STAN_DEBUG_PATH_TAYLOR_APPX false || STAN_DEBUG_PATH_ALL
#define STAN_DEBUG_PATH_ELBO_DRAWS false || STAN_DEBUG_PATH_ALL
#define STAN_DEBUG_PATH_CURVE_CHECK false || STAN_DEBUG_PATH_ALL
#define STAN_DEBUG_PATH_BEST_ELBO false || STAN_DEBUG_PATH_ALL
#define STAN_DEBUG_PATH_RNORM_DRAWS false || STAN_DEBUG_PATH_ALL
#define STAN_DEBUG_PATH_ITERS                                      \
  STAN_DEBUG_PATH_ALL || STAN_DEBUG_PATH_POST_LBFGS                \
      || STAN_DEBUG_PATH_TAYLOR_APPX || STAN_DEBUG_PATH_ELBO_DRAWS \
      || STAN_DEBUG_PATH_CURVE_CHECK || STAN_DEBUG_PATH_BEST_ELBO  \
      || STAN_DEBUG_PATH_RNORM_DRAWS

namespace stan {
namespace services {
namespace optimize {

template <typename T1, typename T2>
inline auto crossprod(T1&& x, T2&& y) {
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
inline auto tcrossprod(T1&& x, T2&& y) {
  return x * y.transpose();
}

template <typename T1>
inline Eigen::MatrixXd tcrossprod(T1&& x) {
  return Eigen::MatrixXd(x.rows(), x.rows())
      .setZero()
      .selfadjointView<Eigen::Lower>()
      .rankUpdate(x);
}

template <typename EigVec1, typename EigVec2>
inline Eigen::MatrixXd std_vec_matrix_times_diagonal(
    const std::vector<EigVec1>& y_buff, const EigVec2& alpha) {
  Eigen::MatrixXd ret(y_buff.size(), alpha.size());
  for (Eigen::Index i = 0; i < y_buff.size(); ++i) {
    ret.row(i) = y_buff[i].array() * alpha.array();
  }
  return ret;
}

template <typename EigVec1, typename EigVec2>
inline Eigen::VectorXd std_vec_matrix_crossprod_vector(
    const std::vector<EigVec1>& y_buff, const EigVec2& x) {
  Eigen::VectorXd ret(y_buff[0].size());
  ret.setZero();
  for (Eigen::Index i = 0; i < y_buff.size(); ++i) {
    ret.noalias() += y_buff[i] * x[i];
  }
  return ret;
}

template <typename EigVec1, typename EigVec2>
inline Eigen::MatrixXd std_vec_matrix_mul_vector(
    const std::vector<EigVec1>& y_buff, const EigVec2& alpha) {
  Eigen::VectorXd ret(y_buff.size());
  for (Eigen::Index i = 0; i < y_buff.size(); ++i) {
    ret(i) = y_buff[i].dot(alpha);
  }
  return ret;
}

inline bool is_nan(double x) noexcept {
  return x == std::numeric_limits<double>::quiet_NaN();
}

inline bool is_infinite(double x) noexcept {
  return x == std::numeric_limits<double>::infinity();
}

template <typename EigMat, stan::require_matrix_t<EigMat>* = nullptr>
inline Eigen::Array<bool, -1, 1> check_curve(const EigMat& Yk,
                                             const EigMat& Sk) {
  auto Dk = ((Yk.array()) * Sk.array()).colwise().sum().eval();
  auto thetak = (Yk.array().square().colwise().sum() / Dk).abs().eval();
  if (STAN_DEBUG_PATH_CURVE_CHECK) {
    std::cout << "\n Check Dk: \n" << Dk.transpose() << "\n";
    std::cout << "\n Check thetak: \n" << thetak.transpose() << "\n";
  }
  return ((Dk > 0) && (thetak <= 1e12));
}

/**
 * eq 4.9
 * Gilbert, J.C., Lemaréchal, C. Some numerical experiments with
 * variable-storage quasi-Newton algorithms. Mathematical Programming 45,
 * 407–435 (1989). https://doi.org/10.1007/BF01589113
 */
template <typename EigVec1, typename EigVec2, typename EigVec3>
inline auto form_diag(const EigVec1& alpha_init, const EigVec2& Yk,
                      const EigVec3& Sk) {
  double y_alpha_y = (Yk.dot(alpha_init.asDiagonal() * Yk));
  double y_s = Yk.dot(Sk);
  double s_inv_alpha_s
      = Sk.dot(alpha_init.array().inverse().matrix().asDiagonal() * Sk);
  return y_s
         / (y_alpha_y / alpha_init.array() + Yk.array().square()
            - (y_alpha_y / s_inv_alpha_s)
                  * (Sk.array() / alpha_init.array()).square());
}

struct taylor_approx_t {
  Eigen::VectorXd x_center;
  double logdetcholHk;
  Eigen::MatrixXd L_approx;
  Eigen::MatrixXd Qk;
  bool use_full;
};

struct elbo_est_t {
  double elbo{-std::numeric_limits<double>::infinity()};
  size_t fn_calls_elbo{0};
  Eigen::MatrixXd repeat_draws;
  Eigen::MatrixXd lp_mat;
};

template <typename EigMat, typename EigVec,
          require_eigen_matrix_dynamic_t<EigMat>* = nullptr>
inline auto gen_draws(EigMat&& u, const taylor_approx_t& taylor_approx,
                      const EigVec& alpha) {
  if (taylor_approx.use_full) {
    return (crossprod(taylor_approx.L_approx, u).colwise()
            + taylor_approx.x_center)
        .eval();
  } else {
    auto u1 = (taylor_approx.Qk.transpose() * u).eval();
    return ((alpha.array().sqrt().matrix().asDiagonal()
             * (taylor_approx.Qk * crossprod(taylor_approx.L_approx, u1)
                + (u - taylor_approx.Qk * u1)))
                .colwise()
            + taylor_approx.x_center)
        .eval();
  }
}

template <typename EigVec1, typename EigVec2,
          require_eigen_vector_t<EigVec1>* = nullptr>
inline auto gen_draws(EigVec1&& u, const taylor_approx_t& taylor_approx,
                      const EigVec2& alpha) {
  if (taylor_approx.use_full) {
    return (crossprod(taylor_approx.L_approx, u) + taylor_approx.x_center)
        .eval();
  } else {
    auto u1 = (taylor_approx.Qk.transpose() * u).eval();
    return ((alpha.array().sqrt().matrix().asDiagonal()
             * (taylor_approx.Qk * crossprod(taylor_approx.L_approx, u1)
                + (u - taylor_approx.Qk * u1)))
            + taylor_approx.x_center)
        .eval();
  }
}

template <Eigen::Index RowsAtCompileTime = -1,
          Eigen::Index ColsAtCompileTime = -1, typename Generator>
inline auto gen_eigen_matrix(Generator&& variate_generator,
                             const Eigen::Index num_params,
                             const Eigen::Index num_samples) {
  return Eigen::Matrix<double, RowsAtCompileTime, ColsAtCompileTime>::
      NullaryExpr(num_params, num_samples,
                  [&variate_generator]() { return variate_generator(); });
}
template <bool ReturnElbo = true, typename F, typename Generator,
          typename EigVec, typename SamplePkg, typename Logger>
inline elbo_est_t est_approx_draws(F&& fn, Generator&& generator,
                                   const SamplePkg& taylor_approx,
                                   size_t num_samples, const EigVec& alpha,
                                   Logger&& logger, int num_eval_attempts) {
  const auto num_params = taylor_approx.x_center.size();
  size_t fn_calls_elbo = 0;
  Eigen::MatrixXd uniform_samps
      = gen_eigen_matrix(generator, num_params, num_samples);
  auto approx_samples = gen_draws(uniform_samps, taylor_approx, alpha);
  if (STAN_DEBUG_PATH_RNORM_DRAWS) {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                 ", ", ", ", "\n", "", "", " ");
    Eigen::MatrixXd param_vals = approx_samples;
    auto mean_vals = param_vals.rowwise().mean().eval();
    std::cout << "Mean Values: \n"
              << mean_vals.transpose().eval().format(CommaInitFmt) << "\n";
    std::cout << "SD Values: \n"
              << (((param_vals.colwise() - mean_vals)
                       .array()
                       .square()
                       .matrix()
                       .rowwise()
                       .sum()
                       .array()
                   / (param_vals.cols() - 1))
                      .sqrt())
                     .transpose()
                     .eval()
              << "\n";
  }
  Eigen::MatrixXd lp_mat(num_samples, 2);
  Eigen::VectorXd approx_samples_col;
  std::stringstream pathfinder_ss;
  for (Eigen::Index i = 0; i < num_samples; ++i) {
    bool sample_good = false;
    for (size_t fail_trys = 0; fail_trys <= num_eval_attempts; ++fail_trys) {
      approx_samples_col = approx_samples.col(i);
      try {
        lp_mat(i, 1) = fn(approx_samples_col, pathfinder_ss);
        if (!std::isfinite(lp_mat(i, 1))) {
          if (fail_trys == num_eval_attempts) {
            throw std::domain_error(std::string("Approximate estimation failed after ") + std::to_string(num_eval_attempts) + " attempts because the approximated samples returned back log(0) from log_prob");
          }
          uniform_samps.col(i)
              = gen_eigen_matrix<-1, 1>(generator, num_params, 1);
          approx_samples.col(i)
              = gen_draws(uniform_samps.col(i), taylor_approx, alpha);
        } else {
          sample_good = true;
        }
      } catch (const std::exception& e) {
        if (fail_trys == num_eval_attempts) {
          throw std::domain_error(
              std::string("Approximate estimation failed after ")
              + std::to_string(num_eval_attempts)
              + " attempts with final error: \n" + e.what());
        }
        uniform_samps.col(i)
            = gen_eigen_matrix<-1, 1>(generator, num_params, 1);
        approx_samples.col(i)
            = gen_draws(uniform_samps.col(i), taylor_approx, alpha);
      }
      if (pathfinder_ss.str().length() > 0) {
        logger.info(pathfinder_ss);
        pathfinder_ss.str("");
      }
      ++fn_calls_elbo;
      if (sample_good) {
        break;
      }
    }
  }
  //### Divergence estimation ###
  lp_mat.col(0) = (-taylor_approx.logdetcholHk)
                  + -0.5
                        * (uniform_samps.array().square().colwise().sum()
                           + num_params * stan::math::LOG_TWO_PI);
  if (ReturnElbo) {
    double ELBO = ((-lp_mat.col(1)) - lp_mat.col(0)).mean();
    if (STAN_DEBUG_PATH_ELBO_DRAWS) {
      Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                   ", ", ", ", "\n", "", "", " ");
      std::cout
          << "\n Rando Sums: \n"
          << approx_samples.array().square().colwise().sum().eval().format(
                 CommaInitFmt)
          << "\n";
      std::cout << "logdetcholHk: " << taylor_approx.logdetcholHk << "\n";
      std::cout << "ELBO: " << ELBO << "\n";
      std::cout << "repeat_draws: \n"
                << approx_samples.transpose().eval().format(CommaInitFmt)
                << "\n";
      /*std::cout << "random_stuff: \n"
                << std::get<0>(tuple_u).transpose().eval().format(CommaInitFmt)
                << "\n";*/
      std::cout << "lp_approx: \n"
                << lp_mat.col(1).transpose().eval().format(CommaInitFmt)
                << "\n";
      std::cout << "fn_call: \n"
                << lp_mat.col(0).transpose().eval().format(CommaInitFmt)
                << "\n";
      Eigen::MatrixXd param_vals = approx_samples;
      auto mean_vals = param_vals.rowwise().mean().eval();
      std::cout << "Mean Values: \n"
                << mean_vals.transpose().eval().format(CommaInitFmt) << "\n";
      std::cout << "SD Values: \n"
                << (((param_vals.colwise() - mean_vals)
                         .array()
                         .square()
                         .matrix()
                         .rowwise()
                         .sum()
                         .array()
                     / (param_vals.cols() - 1))
                        .sqrt())
                       .transpose()
                       .eval()
                << "\n";
    }
    return elbo_est_t{ELBO, fn_calls_elbo, std::move(approx_samples),
                      std::move(lp_mat)};
  } else {
    return elbo_est_t{-std::numeric_limits<double>::infinity(), fn_calls_elbo,
                      std::move(approx_samples), std::move(lp_mat)};
  }
}

template <typename EigVec, typename Buff, typename AlphaVec, typename DkVec,
          typename InvMat>
inline auto construct_taylor_approximation_full(
    const Buff& Ykt_mat, const AlphaVec& alpha, const DkVec& Dk,
    const InvMat& ninvRST, const EigVec& point_est, const EigVec& grad_est) {
  if (STAN_DEBUG_PATH_TAYLOR_APPX) {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n",
                                 "", "", " ");
    std::cout << "---Full---\n";

    std::cout << "Alpha: \n" << alpha.format(CommaInitFmt) << "\n";
    std::cout << "ninvRST: \n" << ninvRST.format(CommaInitFmt) << "\n";
    std::cout << "Dk: \n" << Dk.format(CommaInitFmt) << "\n";
    std::cout << "Point: \n" << point_est.format(CommaInitFmt) << "\n";
    std::cout << "grad: \n" << grad_est.format(CommaInitFmt) << "\n";
  }
  Eigen::MatrixXd y_tcrossprod_alpha = tcrossprod(std_vec_matrix_times_diagonal(
      Ykt_mat, alpha.array().sqrt().matrix().eval()));
  y_tcrossprod_alpha += Dk.asDiagonal();
  const auto dk_min_size
      = std::min(y_tcrossprod_alpha.rows(), y_tcrossprod_alpha.cols());
  Eigen::MatrixXd y_mul_alpha = std_vec_matrix_times_diagonal(Ykt_mat, alpha);
  Eigen::MatrixXd Hk = crossprod(y_mul_alpha, ninvRST)
                       + crossprod(ninvRST, y_mul_alpha)
                       + crossprod(ninvRST, y_tcrossprod_alpha * ninvRST);
  Hk += alpha.asDiagonal();
  Eigen::MatrixXd L_hk = Hk.llt().matrixL().transpose();
  double logdetcholHk = L_hk.diagonal().array().abs().log().sum();

  Eigen::VectorXd x_center = point_est - Hk * grad_est;
  if (STAN_DEBUG_PATH_TAYLOR_APPX) {
    std::cout << "---Full---\n";
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n",
                                 "", "", " ");
    std::cout << "Hk: " << Hk.format(CommaInitFmt) << "\n";
    std::cout << "L_approx: \n" << L_hk.format(CommaInitFmt) << "\n";
    std::cout << "logdetcholHk: \n" << logdetcholHk << "\n";
    std::cout << "x_center: \n" << x_center.format(CommaInitFmt) << "\n";
  }
  return taylor_approx_t{std::move(x_center), logdetcholHk, std::move(L_hk),
                         Eigen::MatrixXd(0, 0), true};
}

template <typename EigVec, typename Buff, typename AlphaVec, typename DkVec,
          typename InvMat>
inline auto construct_taylor_approximation_sparse(
    const Buff& Ykt_mat, const AlphaVec& alpha, const DkVec& Dk,
    const InvMat& ninvRST, const EigVec& point_est, const EigVec& grad_est) {
  const Eigen::Index current_history_size = Ykt_mat.size();
  Eigen::MatrixXd y_mul_sqrt_alpha = std_vec_matrix_times_diagonal(
      Ykt_mat, alpha.array().sqrt().matrix().eval());
  Eigen::MatrixXd Wkbart(Ykt_mat.size() + ninvRST.rows(), alpha.size());
  Wkbart.topRows(Ykt_mat.size()) = y_mul_sqrt_alpha;
  Wkbart.bottomRows(ninvRST.rows())
      = ninvRST * alpha.array().inverse().sqrt().matrix().asDiagonal();

  if (STAN_DEBUG_PATH_TAYLOR_APPX) {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n",
                                 "", "", " ");
    std::cout << "---Sparse---\n";
    std::cout << "Wkbar: \n" << Wkbart.format(CommaInitFmt) << "\n";
  }
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
  const auto min_size = std::min(Wkbart.rows(), Wkbart.cols());
  Eigen::HouseholderQR<Eigen::Ref<decltype(Wkbart)>> qr(Wkbart);
  Eigen::MatrixXd Rkbar = qr.matrixQR().topLeftCorner(min_size, Wkbart.cols());
  Rkbar.triangularView<Eigen::StrictlyLower>().setZero();
  Eigen::MatrixXd Qk
      = qr.householderQ() * Eigen::MatrixXd::Identity(Wkbart.rows(), min_size);
  Eigen::MatrixXd L_approx
      = (Rkbar * Mkbar * Rkbar.transpose()
         + Eigen::MatrixXd::Identity(Rkbar.rows(), Rkbar.rows()))
            .llt()
            .matrixL()
            .transpose();
  double logdetcholHk = L_approx.diagonal().array().abs().log().sum()
                        + 0.5 * alpha.array().log().sum();
  Eigen::VectorXd ninvRSTg = ninvRST * grad_est;
  Eigen::VectorXd alpha_mul_grad = (alpha.array() * grad_est.array()).matrix();

  Eigen::VectorXd x_center
      = point_est
        - (alpha_mul_grad
           + (alpha.array()
              * std_vec_matrix_crossprod_vector(Ykt_mat, ninvRSTg).array())
                 .matrix()
           + crossprod(ninvRST,
                       std_vec_matrix_mul_vector(Ykt_mat, alpha_mul_grad))
           + crossprod(ninvRST, y_tcrossprod_alpha * ninvRSTg));

  if (STAN_DEBUG_PATH_TAYLOR_APPX) {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n",
                                 "", "", " ");
    std::cout << "Full QR: \n" << qr.matrixQR().format(CommaInitFmt) << "\n";
    std::cout << "Alpha: \n" << alpha.format(CommaInitFmt) << "\n";
    std::cout << "Qk: \n" << Qk.format(CommaInitFmt) << "\n";
    std::cout << "L_approx: \n" << L_approx.format(CommaInitFmt) << "\n";
    std::cout << "logdetcholHk: \n" << logdetcholHk << "\n";
    std::cout << "Mkbar: \n" << Mkbar.format(CommaInitFmt) << "\n";
    std::cout << "Decomp Wkbar: \n" << Wkbart.format(CommaInitFmt) << "\n";
    std::cout << "x_center: \n" << x_center.format(CommaInitFmt) << "\n";
    std::cout << "NinvRST: " << ninvRST.format(CommaInitFmt) << "\n";
    std::cout << "ninvRSTg: \n" << ninvRSTg.format(CommaInitFmt) << "\n";
    std::cout << "Rkbar: " << Rkbar.format(CommaInitFmt) << "\n";
  }
  return taylor_approx_t{std::move(x_center), logdetcholHk, std::move(L_approx),
                         std::move(Qk), false};
}

template <typename EigVec, typename Buff, typename AlphaVec, typename DkVec,
          typename InvMat>
inline auto construct_taylor_approximation(
    const Buff& Ykt_mat, const AlphaVec& alpha, const DkVec& Dk,
    const InvMat& ninvRST, const EigVec& point_est, const EigVec& grad_est) {
  // If twice the current history size is larger than the number of params
  // use a sparse approximation
  if (2 * Ykt_mat.size() >= Ykt_mat[0].size()) {
    return construct_taylor_approximation_full(Ykt_mat, alpha, Dk, ninvRST,
                                               point_est, grad_est);
  } else {
    return construct_taylor_approximation_sparse(Ykt_mat, alpha, Dk, ninvRST,
                                                 point_est, grad_est);
  }
}

template <bool ReturnLpSamples, typename EigMat, typename EigVec,
          std::enable_if_t<ReturnLpSamples>* = nullptr>
inline auto ret_pathfinder(int return_code, EigVec&& lp_ratio,
                           EigMat&& samples) {
  return std::make_tuple(return_code, std::forward<EigVec>(lp_ratio),
                         std::forward<EigMat>(samples));
}

template <bool ReturnLpSamples, typename EigMat, typename EigVec,
          std::enable_if_t<!ReturnLpSamples>* = nullptr>
inline auto ret_pathfinder(int return_code, EigVec&& lp_ratio,
                           EigMat&& samples) noexcept {
  return return_code;
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
 * @param[in] num_elbo_draws (K in paper) number of MC draws to evaluate ELBO
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
template <bool ReturnLpSamples = false, class Model, typename DiagnosticWriter,
          typename ParamWriter>
inline auto pathfinder_lbfgs_single(
    Model& model, const stan::io::var_context& init, unsigned int random_seed,
    unsigned int path, double init_radius, int history_size, double init_alpha,
    double tol_obj, double tol_rel_obj, double tol_grad, double tol_rel_grad,
    double tol_param, int num_iterations, bool save_iterations, int refresh,
    callbacks::interrupt& interrupt, int num_elbo_draws, int num_draws,
    int num_eval_attempts, size_t num_threads, callbacks::logger& logger,
    callbacks::writer& init_writer, ParamWriter& parameter_writer,
    DiagnosticWriter& diagnostic_writer) {
  const auto start_optim_time = std::chrono::steady_clock::now();
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
  using Optimizer
      = stan::optimization::BFGSLineSearch<Model, lbfgs_update_t, true>;
  Optimizer lbfgs(model, cont_vector, disc_vector, std::move(ls_opts),
                  std::move(conv_opts), std::move(lbfgs_update), &lbfgs_ss);

  std::string initial_msg("Initial log joint probability = "
                          + std::to_string(lbfgs.logp()));
  const std::string path_num("Path: [" + std::to_string(path) + "] ");
  logger.info(path_num + initial_msg);

  std::vector<std::string> names;
  model.constrained_param_names(names, true, true);
  names.push_back("lp_approx__");
  names.push_back("lp__");
  parameter_writer(names);
  diagnostic_writer(names);
  /*
   * 2. Run L-BFGS to return optimization path for parameters, gradients of
   * objective function, and factorization of covariance estimation
   */
  int ret = 0;
  /*
  Eigen::MatrixXd param_mat(param_size, num_iterations + 1);
  Eigen::MatrixXd grad_mat(param_size, num_iterations + 1);
  */
  std::vector<Eigen::VectorXd> param_vecs;
  param_vecs.reserve(num_iterations);
  std::vector<Eigen::VectorXd> grad_vecs;
  grad_vecs.reserve(num_iterations);
  {
    std::vector<double> g1;
    double blah = stan::model::log_prob_grad<true, true>(model, cont_vector,
                                                         disc_vector, g1);
    param_vecs.emplace_back(
        Eigen::Map<Eigen::VectorXd>(cont_vector.data(), param_size));
    grad_vecs.emplace_back(Eigen::Map<Eigen::VectorXd>(g1.data(), param_size));
  }
  int param_cols_filled = 0;
  while (ret == 0) {
    std::stringstream msg;
    interrupt();
    ret = lbfgs.step();
    double lp = lbfgs.logp();
    if (refresh > 0
        && (ret != 0 || !lbfgs.note().empty() || lbfgs.iter_num() == 0
            || ((lbfgs.iter_num() + 1) % refresh == 0))) {
      std::stringstream msg;
      msg << path_num +
          "    Iter"
          "      log prob"
          "        ||dx||"
          "      ||grad||"
          "       alpha"
          "      alpha0"
          "  # evals"
          "  Notes \n";
      msg << path_num << " " << std::setw(7) << lbfgs.iter_num() << " ";
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
      logger.info(msg.str());
    }

    if (lbfgs_ss.str().length() > 0) {
      logger.info(lbfgs_ss);
      lbfgs_ss.str("");
    }
    /*
     * If the retcode is -1 then linesearch failed even with a hessian reset
     * so the current vals and grads are the same as the previous iter
     * and we are exiting
     */
    if (likely(ret != -1)) {
      param_vecs.emplace_back(lbfgs.curr_x());
      grad_vecs.emplace_back(lbfgs.curr_g());
      ++param_cols_filled;
    }
    if (msg.str().length() > 0) {
      logger.info(msg);
    }
  }
  int return_code;
  if (ret >= 0) {
    logger.info("Optimization terminated normally: ");
    return_code = error_codes::OK;
  } else {
    logger.info("Optimization terminated with error: ");
    logger.info(
        "Stan will still attempt pathfinder but may fail or produce incorrect "
        "results.");
    return_code = error_codes::SOFTWARE;
  }
  logger.info("  " + lbfgs.get_code_string(ret));

  const auto end_optim_time = std::chrono::steady_clock::now();
  const double optim_delta_time
      = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_optim_time - start_optim_time)
            .count()
        / 1000.0;
  const auto start_pathfinder_time = std::chrono::steady_clock::now();
  // 3. Setup param and grad differences and updates of diagonal hessian (alpha)
  /*
  Eigen::MatrixXd Ykt_diff = grad_mat.middleCols(1, param_cols_filled - 1)
                             - grad_mat.leftCols(param_cols_filled - 1);
  Eigen::MatrixXd Skt_diff = param_mat.middleCols(1, param_cols_filled - 1)
                             - param_mat.leftCols(param_cols_filled - 1);
  */
  Eigen::MatrixXd Ykt_diff(param_size, param_cols_filled);
  Eigen::MatrixXd Skt_diff(param_size, param_cols_filled);
  for (Eigen::Index i = 0; i < param_cols_filled; ++i) {
    Ykt_diff.col(i) = param_vecs[i + 1] - param_vecs[i];
    Skt_diff.col(i) = grad_vecs[i + 1] - grad_vecs[i];
  }
  const auto diff_size = Ykt_diff.cols();
  Eigen::MatrixXd alpha_mat(param_size, diff_size);
  Eigen::Matrix<bool, -1, 1> check_curve_vec = check_curve(Ykt_diff, Skt_diff);
  alpha_mat.col(0).setOnes();
  for (Eigen::Index iter = 1; iter < diff_size; iter++) {
    if (check_curve_vec[iter]) {
      alpha_mat.col(iter) = form_diag(alpha_mat.col(iter - 1),
                                      Ykt_diff.col(iter), Skt_diff.col(iter));
    } else {
      alpha_mat.col(iter) = alpha_mat.col(iter - 1);
    }
  }
  std::mutex update_best_mutex;
  if (STAN_DEBUG_PATH_POST_LBFGS) {
    std::lock_guard<std::mutex> guard(update_best_mutex);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n",
                                 "", "", " ");
    std::cout << "\n num_params: " << param_size << "\n";
    std::cout << "\n num_elbo_params: " << num_elbo_draws << "\n";
    std::cout << "\n param_cols_filled: " << param_cols_filled << "\n";
    std::cout << "\n Alpha mat: "
              << alpha_mat.transpose().eval().format(CommaInitFmt) << "\n";
    std::cout << "\n Ykt_diff mat: "
              << Ykt_diff.transpose().eval().format(CommaInitFmt) << "\n";
    /*
std::cout << "\n grad mat: "
    << grad_mat.leftCols(param_cols_filled)
           .transpose()
           .eval()
           .format(CommaInitFmt)
    << "\n";
    */
    std::cout << "\n Skt_diff mat: "
              << Skt_diff.transpose().eval().format(CommaInitFmt) << "\n";
    /*
std::cout << "\n param mat: "
    << param_mat.leftCols(param_cols_filled)
           .transpose()
           .eval()
           .format(CommaInitFmt)
    << "\n";
    */
  }
  auto fn = [&model](auto&& u, auto&& streamer) {
    return -model.template log_prob<false, true>(u, &streamer);
  };
  // NOTE: We always push the first one no matter what
  check_curve_vec[0] = true;
  std::vector<boost::ecuyer1988> rng_vec;
  rng_vec.reserve(diff_size);
  for (Eigen::Index i = 0; i < diff_size; i++) {
    rng_vec.emplace_back(
        util::create_rng<boost::ecuyer1988>(random_seed, path + i));
  }
  Eigen::MatrixXd elbo_mat;
  if (STAN_DEBUG_PATH_BEST_ELBO) {
    elbo_mat = Eigen::MatrixXd(diff_size, 2);
    for (int i = 0; i < elbo_mat.rows(); ++i) {
      elbo_mat(i, 0) = i;
    }
  }
  Eigen::Index best_E = -1;
  elbo_est_t elbo_fit_best;
  taylor_approx_t taylor_approx_best;
  size_t winner = 0;
  tbb::parallel_for(
      tbb::blocked_range<Eigen::Index>(0, diff_size),
      [&](tbb::blocked_range<Eigen::Index> r) {
        for (int iter = r.begin(); iter < r.end(); ++iter) {
          std::string iter_msg(path_num + "Iter: [" + std::to_string(iter)
                               + "] ");
          if (STAN_DEBUG_PATH_ITERS) {
            std::cout << "\n------------ Iter: " << iter << "------------\n";
          }
          auto alpha = alpha_mat.col(iter);
          std::vector<Eigen::Index> ys_cols;
          ys_cols.reserve(history_size);
          {
            for (Eigen::Index end_iter = iter; end_iter >= 0; --end_iter) {
              if (check_curve_vec[end_iter]) {
                ys_cols.push_back(end_iter);
              }
              if (ys_cols.size() == history_size) {
                break;
              }
            }
          }
          const auto current_history_size = ys_cols.size();
          std::vector<decltype(Ykt_diff.col(0))> Ykt_h;
          Ykt_h.reserve(current_history_size);
          Eigen::MatrixXd Skt_mat(Skt_diff.rows(), current_history_size);
          for (Eigen::Index i = 0; i < current_history_size; ++i) {
            Ykt_h.push_back(Ykt_diff.col(ys_cols[i]));
            Skt_mat.col(i) = Skt_diff.col(ys_cols[i]);
          }
          Eigen::VectorXd Dk(current_history_size);
          for (Eigen::Index i = 0; i < current_history_size; i++) {
            Dk.coeffRef(i) = Ykt_h[i].dot(Skt_mat.col(i));
          }
          Eigen::MatrixXd Rk = Eigen::MatrixXd::Zero(current_history_size,
                                                     current_history_size);
          for (Eigen::Index s = 0; s < current_history_size; s++) {
            for (Eigen::Index i = 0; i <= s; i++) {
              Rk.coeffRef(i, s) = Skt_mat.col(i).dot(Ykt_h[s]);
            }
          }
          Eigen::MatrixXd ninvRST;
          {
            Skt_mat.transposeInPlace();
            if (STAN_DEBUG_PATH_ITERS) {
              std::cout << "\nRk: \n" << Rk << "\n";
              std::cout << "\nSkt: \n" << Skt_mat << "\n";
            }
            Rk.triangularView<Eigen::Upper>().solveInPlace(Skt_mat);
            ninvRST = std::move(-Skt_mat);
            if (STAN_DEBUG_PATH_ITERS) {
              std::cout << "\nninvRST: \n" << ninvRST << "\n";
            }
          }
          /**
           * 3a(1). Run BFGS-Sample to get pieces need for sampling.
           */
          taylor_approx_t taylor_appx_tuple = construct_taylor_approximation(
              Ykt_h, alpha, Dk, ninvRST, param_vecs[iter + 1],
              grad_vecs[iter + 1]);

          /**
           * 3a(1). Get `num_elbo_draws` draws from normal
           * approximation and log density of draws in the approximate normal
           * distribution
           */
          //  4. Find $l \in L$ that maximizes ELBO $l^* = arg max_l ELBO^(l)$.
          boost::variate_generator<boost::ecuyer1988&,
                                   boost::normal_distribution<>>
              rand_unit_gaus(rng_vec[iter], boost::normal_distribution<>());
          elbo_est_t elbo_est;
          try {
            elbo_est = est_approx_draws<true>(fn, rand_unit_gaus,
                                              taylor_appx_tuple, num_elbo_draws,
                                              alpha, logger, num_eval_attempts);
          } catch (const std::exception& e) {
            logger.info(
                std::string("ELBO estimation failed for LBFGS iteration "
                            + std::to_string(iter) + " with error: ")
                + e.what());
          }
          if (refresh > 0
              && (lbfgs.iter_num() == 0
                  || ((lbfgs.iter_num() + 1) % refresh == 0))) {
            logger.info(iter_msg + ": ELBO (" + std::to_string(elbo_est.elbo)
                        + ")");
          }
          if (STAN_DEBUG_PATH_BEST_ELBO) {
            elbo_mat(iter, 1) = elbo_est.elbo;
          }
          {
            if (STAN_DEBUG_PATH_BEST_ELBO) {
              std::cout << "elbo best: " << elbo_fit_best.elbo << "\n";
            }
            std::lock_guard<std::mutex> guard(update_best_mutex);
            if (elbo_est.elbo > elbo_fit_best.elbo) {
              elbo_fit_best = std::move(elbo_est);
              taylor_approx_best = std::move(taylor_appx_tuple);
              best_E = iter;
            }
          }
        }
      });
  if (STAN_DEBUG_PATH_BEST_ELBO) {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n",
                                 "", "", " ");
    std::cout << "ELBOs: \n" << elbo_mat.format(CommaInitFmt) << "\n";
    std::cout << "Winner: " << best_E << "\n";
    std::cout << "optim vals: \n" << param_vecs[best_E + 1] << "\n";
  }
  if (best_E == -1) {
    logger.info(
        "Pathfinder Failure: None of the LBFGS iterations completed "
        "successfully");
    return ret_pathfinder<ReturnLpSamples>(-1, Eigen::VectorXd(0),
                                           Eigen::MatrixXd(0, 0));
  }
  Eigen::MatrixXd constrainted_draws_mat(names.size(), num_draws);
  Eigen::VectorXd lp_ratio(num_draws);
  {
    lp_ratio.head(num_elbo_draws)
        = -elbo_fit_best.lp_mat.col(1) - elbo_fit_best.lp_mat.col(0);
    constrainted_draws_mat.block(names.size() - 2, 0, 2, num_elbo_draws)
        = elbo_fit_best.lp_mat.transpose();
    tbb::parallel_for(
        tbb::blocked_range<Eigen::Index>(0, num_elbo_draws, 512),
        [&](tbb::blocked_range<Eigen::Index> r) {
          Eigen::VectorXd unconstrained_draws;
          Eigen::VectorXd constrained_draws;
          for (Eigen::Index i = r.begin(); i < r.end(); ++i) {
            //    for (Eigen::Index i = 0; i < num_elbo_draws; ++i) {
            unconstrained_draws = elbo_fit_best.repeat_draws.col(i);
            model.write_array(rng, unconstrained_draws, constrained_draws);
            constrainted_draws_mat.col(i).head(names.size() - 2)
                = constrained_draws;
          }
        });
  }
  if ((num_draws - num_elbo_draws) > 0) {
    boost::variate_generator<boost::ecuyer1988&, boost::normal_distribution<>>
        rand_unit_gaus(rng_vec[best_E], boost::normal_distribution<>());

    tbb::parallel_for(
        tbb::blocked_range<Eigen::Index>(num_elbo_draws, num_draws, 256),
        [&](tbb::blocked_range<Eigen::Index> r) {
          Eigen::VectorXd unconstrained_draws;
          Eigen::VectorXd constrained_draws;
          elbo_est_t draws_tuple;
          try {
            draws_tuple = est_approx_draws<false>(
                fn, rand_unit_gaus, taylor_approx_best, r.size(),
                alpha_mat.col(best_E), logger, num_eval_attempts);
          } catch (const std::exception& e) {
            logger.info(
                std::string("Final sampling approximation failed with error: ")
                + e.what());
          }
          lp_ratio.segment(static_cast<Eigen::Index>(r.begin()), r.size())
              = -draws_tuple.lp_mat.col(1) - draws_tuple.lp_mat.col(0);
          constrainted_draws_mat.block(names.size() - 2,
                                       static_cast<Eigen::Index>(r.begin()), 2,
                                       r.size())
              = draws_tuple.lp_mat.transpose();
          for (Eigen::Index i = r.begin(), idx = 0; i < r.end(); ++i, ++idx) {
            unconstrained_draws = draws_tuple.repeat_draws.col(idx);
            model.write_array(rng, unconstrained_draws, constrained_draws);
            constrainted_draws_mat.col(i).head(names.size() - 2)
                = constrained_draws;
          }
        });
  }
  parameter_writer(constrainted_draws_mat);
  auto end_pathfinder_time = std::chrono::steady_clock::now();
  double pathfinder_delta_time
      = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_pathfinder_time - start_pathfinder_time)
            .count()
        / 1000.0;
  parameter_writer();
  const auto time_header = std::string("Elapsed Time: ");
  std::string optim_time_str
      = time_header + std::to_string(optim_delta_time) + " seconds (lbfgs)";
  parameter_writer(optim_time_str);

  std::string pathfinder_time_str = std::string(time_header.size(), ' ')
                                    + std::to_string(pathfinder_delta_time)
                                    + " seconds (Pathfinder)";
  parameter_writer(pathfinder_time_str);

  std::string total_time_str
      = std::string(time_header.size(), ' ')
        + std::to_string(optim_delta_time + pathfinder_delta_time)
        + " seconds (Total)";
  parameter_writer(total_time_str);

  parameter_writer();

  return ret_pathfinder<ReturnLpSamples>(0, std::move(lp_ratio),
                                         std::move(constrainted_draws_mat));
}

}  // namespace optimize
}  // namespace services
}  // namespace stan
#endif
