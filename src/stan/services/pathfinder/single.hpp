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
#define STAN_DEBUG_PATH_ITERS STAN_DEBUG_PATH_ALL || STAN_DEBUG_PATH_POST_LBFGS || STAN_DEBUG_PATH_TAYLOR_APPX || STAN_DEBUG_PATH_ELBO_DRAWS || STAN_DEBUG_PATH_CURVE_CHECK || STAN_DEBUG_PATH_BEST_ELBO

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
  double elbo;
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
                         Eigen::MatrixXd param_vals = u2;
                         Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "\n", "",
                                                      "", " ");
//                         std::cout << "\n\n RANDO:  \n\n" << u.format(CommaInitFmt) << "\n\n";
/*
                         auto mean_vals = param_vals.rowwise().mean().eval();
                         std::cout << "Mean Values: \n" << mean_vals.transpose().eval().format(CommaInitFmt) << "\n";
                         std::cout << "SD Values: \n"
                                   << (((param_vals.colwise() - mean_vals)
                                           .array()
                                           .square()
                                           .matrix()
                                           .rowwise()
                                           .sum()
                                           .array()
                                       / (param_vals.cols() - 1))
                                          .sqrt()).transpose().eval()
                                   << "\n";
*/
    return std::make_tuple(std::move(u), std::move(u2));
  } else {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                                 "", " ");
//    std::cout << "\n Qk: \n" << taylor_approx.Qk.format(CommaInitFmt) << "\n";
//    std::cout << "\n Qk rows:" << taylor_approx.Qk.rows() << " cols:" << taylor_approx.Qk.cols() << "\n";
//    std::cout << "\n u rows: " << u.rows() << " cols:" << u.cols() << "\n";

//    std::cout << "\n u: \n" << u << "\n";
    Eigen::MatrixXd u1 = taylor_approx.Qk.transpose() * u;
  //  std::cout << "\n u1: \n" << u1.format(CommaInitFmt) << "\n";
    Eigen::MatrixXd u2
        = (alpha.array().sqrt().matrix().asDiagonal()
           * (taylor_approx.Qk * crossprod(taylor_approx.L_approx, u1)
              + (u - taylor_approx.Qk * u1)))
              .colwise()
          + taylor_approx.x_center;
          /*
    std::cout << "\n u2: \n" << u2.format(CommaInitFmt) << "\n";
    Eigen::MatrixXd param_vals = u2;
    auto mean_vals = param_vals.rowwise().mean().eval();
    std::cout << "Mean Values: \n" << mean_vals.transpose().eval().format(CommaInitFmt) << "\n";
    std::cout << "SD Values: \n"
              << (((param_vals.colwise() - mean_vals)
                      .array()
                      .square()
                      .matrix()
                      .rowwise()
                      .sum()
                      .array()
                  / (param_vals.cols() - 1))
                     .sqrt()).transpose().eval()
              << "\n";
              */
    return std::make_tuple(std::move(u), std::move(u2));
  }
}

template <typename F>
inline auto calc_lp_fun(F&& fn, const Eigen::MatrixXd& samples) {
  int draw_ind = 1;
  int fn_calls_DIV = 0;
  Eigen::VectorXd f_test_elbo_draws(samples.cols());
  Eigen::VectorXd u2_col;
  try {
    for (Eigen::Index i = 0; i < samples.cols(); ++i) {
      u2_col = samples.col(i);
      f_test_elbo_draws(i) = fn(u2_col);
      ++fn_calls_DIV;
    }
  } catch (...) {
    // TODO: Actually catch errors
  }
  return f_test_elbo_draws;
}

template <typename SamplePkg, typename F, typename BaseRNG, typename Model>
inline auto est_elbo_draws(const SamplePkg& taylor_approx, size_t num_samples,
                           const Eigen::VectorXd& alpha, F&& fn,
                           BaseRNG&& rnorm, Model& model, Eigen::Index iter = 0) {
  const auto param_size = taylor_approx.x_center.size();
  int draw_ind = 1;
  int fn_calls_DIV = 0;
  auto tuple_u = get_rnorm_and_draws(rnorm, taylor_approx, alpha);
  Eigen::VectorXd f_test_elbo_draws = Eigen::VectorXd::Zero(num_samples);
  Eigen::VectorXd u2_col;
  /*
  auto fn1 = [&model](auto&& u) {
    return model.template log_prob<false, false>(u, 0);
  };
  auto fn2 = [&model](auto&& u) {
    return model.template log_prob<false, true>(u, 0);
  };
  auto fn3 = [&model](auto&& u) {
    return model.template log_prob<true, false>(u, 0);
  };
  auto fn4 = [&model](auto&& u) {
    return model.template log_prob<true, true>(u, 0);
  };
  */
  try {
    for (Eigen::Index i = 0; i < num_samples; ++i) {
      u2_col = std::get<1>(tuple_u).col(i);
      f_test_elbo_draws(i) = -fn(u2_col);
      /*
      double check1 = fn1(u2_col);
      double check2 = fn2(u2_col);
      double check3 = fn3(u2_col);
      double check4 = fn4(u2_col);
      if (iter == 0) {
        std::cout << "\nSample [" << i << "] u2: \n" << u2_col << "\n";
        std::cout << "lp: " << f_test_elbo_draws(i) << "\n";
        std::cout << "lp1: " << check1 << "\n";
        std::cout << "lp2: " << check2 << "\n";
        std::cout << "lp3: " << check3 << "\n";
        std::cout << "lp4: " << check4 << "\n";

      }
      */
      ++fn_calls_DIV;
    }
  } catch (...) {
    std::cout << "\n\n\n YIKES!!!! \n\n\n";
    // TODO: Actually catch errors
  }
  //  std::cout << "\nf_test_elbo_draws: \n" << f_test_elbo_draws << "\n";
  Eigen::VectorXd lp_approx_draws
      = -taylor_approx.logdetcholHk
        - 0.5 * std::get<0>(tuple_u).array().square().colwise().sum()
        - 0.5 * param_size * log(2 * stan::math::pi());
  //### Divergence estimation ###
  double ELBO = (f_test_elbo_draws - lp_approx_draws).mean();
  if (STAN_DEBUG_PATH_ELBO_DRAWS) {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, " ", ", ", "\n", "",
                                 "", " ");
    std::cout << "logdetcholHk: " << taylor_approx.logdetcholHk << "\n";
    std::cout << "ELBO: " << ELBO << "\n";
    std::cout << "repeat_draws: \n"
              << std::get<1>(tuple_u).transpose().eval().format(CommaInitFmt)
              << "\n";
    /*std::cout << "random_stuff: \n"
              << std::get<0>(tuple_u).transpose().eval().format(CommaInitFmt)
              << "\n";*/
    std::cout << "lp_approx_draws: \n"
              << lp_approx_draws.transpose().eval().format(CommaInitFmt) << "\n";
    std::cout << "fn_call: \n"
              << f_test_elbo_draws.transpose().eval().format(CommaInitFmt) << "\n";
  }
  return ELBO;
}

template <typename SamplePkg, typename BaseRNG>
inline auto approximation_samples(const SamplePkg& taylor_approx,
                                  size_t num_samples,
                                  const Eigen::VectorXd& alpha,
                                  BaseRNG&& rnorm) {
  const Eigen::Index num_params = taylor_approx.x_center.size();
  auto tuple_u = get_rnorm_and_draws(rnorm, taylor_approx, alpha);
  auto&& u2 = std::get<1>(tuple_u);
  // TODO: Inline this on the bottom row
  Eigen::VectorXd lp_approx_draws
      = -taylor_approx.logdetcholHk
        - 0.5 * std::get<0>(std::move(tuple_u)).array().square().colwise().sum()
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
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", " ");

  if (STAN_DEBUG_PATH_TAYLOR_APPX) {
    std::cout << "---Full---\n";

    std::cout << "Alpha: \n" << alpha.format(CommaInitFmt) << "\n";
    std::cout << "ninvRST: \n" << ninvRST.format(CommaInitFmt) << "\n";
    std::cout << "Dk: \n" << Dk.format(CommaInitFmt) << "\n";
    std::cout << "Point: \n" << point_est.format(CommaInitFmt) << "\n";
    std::cout << "grad: \n" << grad_est.format(CommaInitFmt) << "\n";
  }
  Eigen::MatrixXd y_tcrossprod_alpha
      = tcrossprod(std_vec_matrix_times_diagonal(
            Ykt_mat, alpha.array().sqrt().matrix().eval()));
  y_tcrossprod_alpha += Dk.asDiagonal();
   //std::cout << "y_tcrossprod_alpha: \n" << y_tcrossprod_alpha << "\n";
  const auto dk_min_size
      = std::min(y_tcrossprod_alpha.rows(), y_tcrossprod_alpha.cols());
  //y_tcrossprod_alpha += Dk.head(dk_min_size).asDiagonal();
  // std::cout << "y_tcrossprod_alpha2: \n" << y_tcrossprod_alpha << "\n";

  Eigen::MatrixXd y_mul_alpha = std_vec_matrix_times_diagonal(Ykt_mat, alpha);
  Eigen::MatrixXd step1 = crossprod(y_mul_alpha, ninvRST);
  Eigen::MatrixXd step2 = crossprod(ninvRST, y_mul_alpha);
  Eigen::MatrixXd step3 = crossprod(ninvRST, y_tcrossprod_alpha * ninvRST);
  /*
  std::cout << "\nstep1: \n" << step1 << "\n";
  std::cout << "\nstep2: \n" << step2 << "\n";
  std::cout << "\nstep3: \n" << step3 << "\n";
  */
  Eigen::MatrixXd Hk = step1 + step2 + step3;
  // std::cout << "Hk: " << Hk.format(CommaInitFmt) << "\n";
  Hk += alpha.asDiagonal();
  // std::cout << "Hk2: " << Hk.format(CommaInitFmt) << "\n";
  Eigen::MatrixXd L_hk = Hk.llt().matrixL().transpose();
  // std::cout << "L_approx: \n" << cholHk.format(CommaInitFmt) << "\n";
  double logdetcholHk = L_hk.diagonal().array().abs().log().sum();

  Eigen::VectorXd x_center = point_est - Hk * grad_est;
  if (STAN_DEBUG_PATH_TAYLOR_APPX) {
    std::cout << "---Full---\n";

    std::cout << "Hk: " << Hk.format(CommaInitFmt) << "\n";
    std::cout << "L_approx: \n" << L_hk.format(CommaInitFmt) << "\n";
    std::cout << "logdetcholHk: \n" << logdetcholHk << "\n";
    std::cout << "x_center: \n" << x_center.format(CommaInitFmt) << "\n";
  }
  return taylor_approx_t{std::move(x_center), logdetcholHk, std::move(L_hk),
                         Eigen::MatrixXd(0, 0), true};
}

template <typename EigVec, typename Buff>
inline auto construct_taylor_approximation_sparse(
    const Buff& Ykt_mat, const Eigen::VectorXd& alpha,
    const Eigen::VectorXd& Dk, const Eigen::MatrixXd& ninvRST,
    const EigVec& point_est, const EigVec& grad_est) {
  const Eigen::Index current_history_size = Ykt_mat.size();
  Eigen::MatrixXd y_mul_sqrt_alpha = std_vec_matrix_times_diagonal(
      Ykt_mat, alpha.array().sqrt().matrix().eval());
  Eigen::MatrixXd Wkbart(Ykt_mat.size() + ninvRST.rows(), alpha.size());
  Wkbart.topRows(Ykt_mat.size()) = y_mul_sqrt_alpha;
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", " ");
  Wkbart.bottomRows(ninvRST.rows())
      = ninvRST * alpha.array().inverse().sqrt().matrix().asDiagonal();

  //std::cout << "Wkbar: \n" << Wkbart.format(CommaInitFmt) << "\n";
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
            .matrixL().transpose();
  double logdetcholHk = L_approx.diagonal().array().abs().log().sum()
                        + 0.5 * alpha.array().log().sum();
  Eigen::VectorXd ninvRSTg = ninvRST * grad_est;
  Eigen::VectorXd alpha_mul_grad = (alpha.array() * grad_est.array()).matrix();
  Eigen::VectorXd x_center_tmp
      = alpha_mul_grad
        + (alpha.array()
           * std_vec_matrix_crossprod_vector(Ykt_mat, ninvRSTg).array())
              .matrix()
        + crossprod(ninvRST, std_vec_matrix_mul_vector(Ykt_mat, alpha_mul_grad))
        + crossprod(ninvRST, y_tcrossprod_alpha * ninvRSTg);

  Eigen::VectorXd x_center = point_est - x_center_tmp;

  if (STAN_DEBUG_PATH_TAYLOR_APPX) {
    std::cout << "---Sparse---\n";
    std::cout << "Full QR: " << qr.matrixQR().format(CommaInitFmt) << "\n";
    std::cout << "Qk: \n" << Qk.format(CommaInitFmt) << "\n";
    std::cout << "L_approx: \n" << L_approx.format(CommaInitFmt) << "\n";
    std::cout << "logdetcholHk: \n" << logdetcholHk << "\n";
    std::cout << "Mkbar: \n" << Mkbar.format(CommaInitFmt) << "\n";
    std::cout << "Decomp Wkbar: \n" << Wkbart.format(CommaInitFmt) << "\n";
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

template <bool ReturnLpSamples, typename EigMat, typename EigVec,
          std::enable_if_t<ReturnLpSamples>* = nullptr>
inline auto ret_pathfinder(int return_code, EigVec&& lp_ratio, EigMat&& samples) {
  return std::make_tuple(return_code, std::forward<EigVec>(lp_ratio), std::forward<EigMat>(samples));
}

template <bool ReturnLpSamples, typename EigMat, typename EigVec,
          std::enable_if_t<!ReturnLpSamples>* = nullptr>
inline auto ret_pathfinder(int return_code, EigVec&& lp_ratio, EigMat&& samples) {
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
template <bool ReturnLpSamples = false, class Model, typename DiagnosticWriter,
          typename ParamWriter>
inline auto pathfinder_lbfgs_single(
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
  using Optimizer = stan::optimization::BFGSLineSearch<Model, lbfgs_update_t, true>;
  Optimizer lbfgs(model, cont_vector, disc_vector, std::move(ls_opts),
                  std::move(conv_opts), std::move(lbfgs_update), &lbfgs_ss);

  std::string initial_msg("Initial log joint probability = " + std::to_string(lbfgs.logp()));
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
  //std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> lbfgs_iters;
  int ret = 0;
  /*
  Eigen::MatrixXd param_mat = given_X;
  Eigen::MatrixXd grad_mat = given_grad;
  int actual_num_iters = given_X.cols() - 1;
  */
  Eigen::MatrixXd param_mat(param_size, num_iterations);
  Eigen::MatrixXd grad_mat(param_size, num_iterations);
  {
    std::vector<double> g1;
    double blah = stan::model::log_prob_grad<true, true>(model, cont_vector,
                                                         disc_vector, g1);

//    lbfgs_iters.emplace_back(
//        Eigen::Map<Eigen::VectorXd>(cont_vector.data(), param_size),
//        Eigen::Map<Eigen::VectorXd>(g1.data(), g1.size()));

    param_mat.col(0)
        = Eigen::Map<Eigen::VectorXd>(cont_vector.data(), param_size);
    grad_mat.col(0) = -Eigen::Map<Eigen::VectorXd>(g1.data(), param_size);
  }
  int actual_num_iters = 0;
  while (ret == 0) {
    std::stringstream msg;
    interrupt();
    ret = lbfgs.step();
    double lp = lbfgs.logp();
    //lbfgs.params_r(cont_vector);

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
      //lbfgs_iters.emplace_back(lbfgs.curr_x(), lbfgs.curr_g());
      ++actual_num_iters;
      param_mat.col(actual_num_iters) = lbfgs.curr_x();
      grad_mat.col(actual_num_iters) = -lbfgs.curr_g();
    }
    if (msg.str().length() > 0) {
      logger.info(msg);
    }
    //std::cout << "\nRet: " << ret << "\n";
  }
  // 3. For each L-BFGS iteration `num_iterations`

  Eigen::MatrixXd Ykt_diff = grad_mat.middleCols(1, actual_num_iters)
                             - grad_mat.leftCols(actual_num_iters);
  Eigen::MatrixXd Skt_diff = param_mat.middleCols(1, actual_num_iters)
                             - param_mat.leftCols(actual_num_iters);
  Eigen::Index best_E;
  double elbo_max = std::numeric_limits<double>::lowest();
  elbo_est_t DIV_fit_best;
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
    std::cout << "\n num_params: " << param_size << "\n";
    std::cout << "\n num_elbo_params: " << num_elbo_draws << "\n";
    std::cout << "\n actual_num_iters: " << actual_num_iters << "\n";
    std::cout << "\n Alpha mat: "
              << alpha_mat.transpose().eval().format(CommaInitFmt) << "\n";
    std::cout << "\n Ykt_diff mat: "
              << Ykt_diff.transpose().eval().format(CommaInitFmt) << "\n";
    std::cout << "\n grad mat: "
              << grad_mat.leftCols(actual_num_iters + 1)
                     .transpose()
                     .eval()
                     .format(CommaInitFmt)
              << "\n";
    std::cout << "\n Skt_diff mat: "
              << Skt_diff.transpose().eval().format(CommaInitFmt) << "\n";
    std::cout << "\n param mat: "
              << param_mat.leftCols(actual_num_iters + 1)
                     .transpose()
                     .eval()
                     .format(CommaInitFmt)
              << "\n";
  }
  std::vector<boost::ecuyer1988> rng_vec;

  auto fn = [&model](auto&& u) {
    /*
    std::vector<double> blah(u.size());
    for (int i = 0; i < u.size(); ++i) {
      blah[i] = u[i];
    }
    std::vector<double> graddd(u.size());
    std::vector<int> par_i;
    return -stan::model::log_prob_grad<true,true>(model, blah, par_i, graddd, 0);
    */
    return -model.template log_prob<true, true>(u, 0);
  };

  // NOTE: We always push the first one no matter what
  std::mutex update_best_mutex;
  actual_num_iters = actual_num_iters > 1 ? actual_num_iters - 1 : actual_num_iters;
  for (Eigen::Index iter = 0; iter < actual_num_iters; iter++) {
    rng_vec.emplace_back(
        util::create_rng<boost::ecuyer1988>(random_seed, path + iter));
  }
  //for (Eigen::Index iter = 0; iter < actual_num_iters; iter++) {

  tbb::parallel_for(
      tbb::blocked_range<int>(0, actual_num_iters),
      [&](tbb::blocked_range<int> r) {
        for (int iter = r.begin(); iter < r.end(); ++iter) {

          std::string iter_msg(path_num + "Iter: [" + std::to_string(iter) + "] ");
          if (STAN_DEBUG_PATH_ITERS) {
            std::cout << "\n------------ Iter: " << iter << "------------\n";
          }
          boost::variate_generator<boost::ecuyer1988&,
                                   boost::normal_distribution<>>
              rand_unit_gaus(rng_vec[iter], boost::normal_distribution<>());
          auto rnorm = [&rand_unit_gaus, num_params = param_size,
                        num_samples = num_elbo_draws]() {
            return Eigen::MatrixXd::NullaryExpr(
                num_params, num_samples,
                [&rand_unit_gaus]() { return rand_unit_gaus(); });
          };
          /*
          auto rnorm = [num_params = param_size,
                        num_samples = num_elbo_draws]() {
            Eigen::MatrixXd blah(num_samples, num_params);
            blah << -0.626453810742332, 0.183643324222082, -0.835628612410047, 1.59528080213779, 0.329507771815361, -0.820468384118015, 0.487429052428485, 0.738324705129217, 0.575781351653492, -0.305388387156356, 1.51178116845085, 0.389843236411431, -0.621240580541804, -2.2146998871775, 1.12493091814311, -0.0449336090152309, -0.0161902630989461, 0.943836210685299, 0.821221195098089, 0.593901321217509, 0.918977371608218, 0.782136300731067, 0.0745649833651906, -1.98935169586337, 0.61982574789471, -0.0561287395290008, -0.155795506705329, -1.47075238389927, -0.47815005510862, 0.417941560199702, 1.35867955152904, -0.102787727342996, 0.387671611559369, -0.0538050405829051, -1.37705955682861, -0.41499456329968, -0.394289953710349, -0.0593133967111857, 1.10002537198388, 0.763175748457544, -0.164523596253587, -0.253361680136508, 0.696963375404737, 0.556663198673657, -0.68875569454952, -0.70749515696212, 0.36458196213683, 0.768532924515416, -0.112346212150228, 0.881107726454215, 0.398105880367068, -0.612026393250771, 0.341119691424425, -1.12936309608079, 1.43302370170104, 1.98039989850586, -0.367221476466509, -1.04413462631653, 0.569719627442413, -0.135054603880824, 2.40161776050478, -0.0392400027331692, 0.689739362450777, 0.0280021587806661, -0.743273208882405, 0.188792299514343, -1.80495862889104, 1.46555486156289, 0.153253338211898, 2.17261167036215, 0.475509528899663, -0.709946430921815, 0.610726353489055, -0.934097631644252, -1.2536334002391, 0.291446235517463, -0.443291873218433, 0.00110535163162413, 0.0743413241516641, -0.589520946188072, -0.568668732818502, -0.135178615123832, 1.1780869965732, -1.52356680042976, 0.593946187628422, 0.332950371213518, 1.06309983727636, -0.304183923634301, 0.370018809916288, 0.267098790772231, -0.54252003099165, 1.20786780598317, 1.16040261569495, 0.700213649514998, 1.58683345454085, 0.558486425565304, -1.27659220845804, -0.573265414236886, -1.22461261489836, -0.473400636439312, -0.620366677224124, 0.0421158731442352, -0.910921648552446, 0.158028772404075, -0.654584643918818, 1.76728726937265, 0.716707476017206, 0.910174229495227, 0.384185357826345, 1.68217608051942, -0.635736453948977, -0.461644730360566, 1.43228223854166, -0.650696353310367, -0.207380743601965, -0.392807929441984, -0.319992868548507, -0.279113302976559, 0.494188331267827, -0.177330482269606, -0.505957462114257, 1.34303882517041, -0.214579408546869, -0.179556530043387, -0.100190741213562, 0.712666307051405, -0.0735644041263263, -0.0376341714670479, -0.681660478755657, -0.324270272246319, 0.0601604404345152, -0.588894486259664, 0.531496192632572, -1.51839408178679, 0.306557860789766, -1.53644982353759, -0.300976126836611, -0.528279904445006, -0.652094780680999, -0.0568967778473925, -1.91435942568001, 1.17658331201856, -1.664972436212, -0.463530401472386, -1.11592010504285, -0.750819001193448, 2.08716654562835, 0.0173956196932517, -1.28630053043433, -1.64060553441858, 0.450187101272656, -0.018559832714638, -0.318068374543844, -0.929362147453702, -1.48746031014148, -1.07519229661568, 1.00002880371391, -0.621266694796823, -1.38442684738449, 1.86929062242358, 0.425100377372448, -0.238647100913033, 1.05848304870902, 0.886422651374936, -0.619243048231147, 2.20610246454047, -0.255027030141015, -1.42449465021281, -0.144399601954219, 0.207538339232345, 2.30797839905936, 0.105802367893711, 0.456998805423414, -0.077152935356531, -0.334000842366544, -0.0347260283112762, 0.787639605630162, 2.07524500865228, 1.02739243876377, 1.2079083983867, -1.23132342155804, 0.983895570053379, 0.219924803660651, -1.46725002909224, 0.521022742648139, -0.158754604716016, 1.4645873119698, -0.766081999604665, -0.430211753928547, -0.926109497377437, -0.17710396143654, 0.402011779486338, -0.731748173119606, 0.830373167981674, -1.20808278630446, -1.04798441280774, 1.44115770684428, -1.01584746530465, 0.411974712317515, -0.38107605110892, 0.409401839650934, 1.68887328620405, 1.58658843344197, -0.330907800682766, -2.28523553529247, 2.49766158983416, 0.667066166765493, 0.5413273359637, -0.0133995231459087, 0.510108422952926, -0.164375831769667, 0.420694643254513, -0.400246743977644, -1.37020787754746, 0.987838267454879, 1.51974502549955, -0.308740569225614, -1.25328975560769, 0.642241305677824, -0.0447091368939791, -1.73321840682484, 0.00213185968026965, -0.630300333928146, -0.340968579860405, -1.15657236263585, 1.80314190791747, -0.331132036391221, -1.60551341225308, 0.197193438739481, 0.263175646405474, -0.985826700409291, -2.88892067167955, -0.640481702565115, 0.570507635920485, -0.05972327604261, -0.0981787440052344, 0.560820728620116, -1.18645863857947, 1.09677704427424, -0.00534402827816569, 0.707310667398079, 1.03410773473746, 0.223480414915304, -0.878707612866019, 1.16296455596733, -2.00016494478548, -0.544790740001725, -0.255670709156989, -0.166121036765006, 1.02046390878411, 0.136221893102778, 0.407167603423836, -0.0696548130129049, -0.247664341619331, 0.69555080661964, 1.1462283572158, -2.40309621489187, 0.572739555245841, 0.374724406778655, -0.425267721556076, 0.951012807576816, -0.389237181718379, -0.284330661799574, 0.857409778079803, 1.7196272991206, 0.270054900937229, -0.42218400978764, -1.18911329485959, -0.33103297887901, -0.939829326510021, -0.258932583118785, 0.394379168221572, -0.851857092023863, 2.64916688109488, 0.156011675665079, 1.13020726745494, -2.28912397984011, 0.741001157195439, -1.31624516045156, 0.919803677609141, 0.398130155451956, -0.407528579269772, 1.32425863017727, -0.70123166924692, -0.580614304240536, -1.00107218102542, -0.668178606753393, 0.945184953373082, 0.433702149545162, 1.00515921767704, -0.390118664053679, 0.376370291774648, 0.244164924486494, -1.42625734238254, 1.77842928747545, 0.134447660933676, 0.765598999157864, 0.955136676908982, -0.0505657014422701, -0.305815419766971, 0.893673702425513, -1.0472981490613, 1.97133738622415, -0.383632106288362, 1.65414530227499, 1.51221269395063, 0.082965733585914, 0.5672209148515, -1.02454847953446, 0.323006503022436, 1.04361245835618, 0.0990784868971982, -0.45413690915436, -0.65578185245044, -0.0359224226225114, 1.0691614606768, -0.483974930301277, -0.121010111327444, -1.29414000382084, 0.494312836014856, 1.30790152011415, 1.49704100940278, 0.814702730897356, -1.86978879020261, 0.48202950412376, 0.456135603301202, -0.353400285829911, 0.170489470947982, -0.864035954126904, 0.679230774015656, -0.327101014653104, -1.56908218514395, -0.367450756170482, 1.36443492906985, -0.334281364731164, 0.732750042209102, 0.946585640197786, 0.00439870432637403, -0.352322305549849, -0.529695509133504, 0.739589225574796, -1.06345741548281, 0.2462108435364, -0.289499366564312, -2.26488935648794, -1.40885045607319, 0.916019328792574, -0.1912789505352, 0.803283216133648, 1.8874744633086, 1.47388118110914, 0.677268492312998, 0.379962686566745, -0.192798426457334, 1.5778917949044, 0.596234109318454, -1.17357694087136, -0.155642534890318, -1.91890982026984, -0.195258846110637, -2.59232766994599, 1.31400216719805, -0.635543001032135, -0.429978838694188, -0.169318332301963, 0.612218173989128, 0.6783401772227, 0.567951972471672, -0.572542603926126, -1.36329125627834, -0.388722244337901, 0.277914132450543, -0.823081121572025, -0.0688409344784646, -1.1676623261298, -0.0083090142156068, 0.128855401597405, -0.145875628461003, -0.163910956736068, 1.76355200278492, 0.762586512418318, 1.11143108073063, -0.923206952829831, 0.164341838427956, 1.15482518709727, -0.0565214245264898, -2.12936064823465, 0.344845762099456, -1.90495544558553, -0.811170153140217, 1.3240043212996, 0.615636849302674, 1.09166895553536, 0.306604861513632, -0.110158762482857, -0.924312773127284, 1.5929137537192, 0.0450105981218803, -0.715128400667883, 0.865223099717138, 1.0744409582779, 1.89565477419858, -0.602997303605094, -0.390867820723137, -0.416222031530192, -0.375657422820391, -0.366630945702358, -0.295677452700886, 1.44182041019987, -0.697538291913041, -0.38816750592136, 0.652536452171517, 1.12477244653513, -0.772110803023928, -0.508086216138287, 0.523620590498017, 1.01775422653797, -0.251164588087978, -1.42999344738861, 1.70912103210088, 1.43506957231162, -0.71037114584988, -0.0650675736555838, -1.75946873536593, 0.569722971819101, 1.61234679820178, -1.63728064710472, -0.779568513201747, -0.641176933750564, -0.681131393571156, -2.03328559561795, 0.500963559247907, -1.53179813996574, -0.0249976392782635, 0.592984721025293, -0.19819542148464, 0.892008392473526, -0.0257150709181537, -0.647660450585665, 0.64635941503464, -0.433832740029588, 1.7726111849785, -0.0182597111630101, 0.852814993600531, 0.205162903244026, -3.00804859892048, -1.36611193132898, -0.424102260144812, 0.236803663745558, -2.34272312035896, 0.961696633380712, -0.60442573385444, -0.752877279362913, -1.55561159254425, -1.45389373800106, 0.0563318360544103, 0.509369406673892, -2.0978829597843, -1.00436197944933, 0.535771722252592, -0.453037084649918, 2.16536850181726, 1.24574667274811, 0.595498034069596, 0.0048844495381889, 0.279360782055259, -0.705906125267626, 0.628017152957251, 1.4802139600857, 1.08342991008328, -0.813244256664091, -1.61887684922359, -0.109655699488241, 0.440889370916978, 1.3509939798086, -1.31860948453356, 0.364384592607448, 0.233499835088518, 1.19395526125905, -0.0279099721295774, -0.357298854504253, -1.14681413611837, -0.517420483593623, -0.362123772578272, 2.35055432578911, 2.44653137569996, -0.166703279484276, -1.04366743906189, -1.97293493407467, 0.514671633438029, -1.09057358373486, 2.28465932554436, -0.885617572657959, 0.111106429745722, 3.81027668071067, -1.10890999794325, 0.307566624421454, -1.10689447222516, 0.34765364882196, -0.873264535062044, 0.0773031227369236, -0.296868642156621, -1.18324224043267, 0.0112926884238996, 0.991601036059367, 1.59396745390374, -1.37271127092917, -0.249610933042319, 1.15942452685555, -1.1142223478436, -2.52850068885039, -0.935902558513312, -0.967239457935565, 0.0474885923277959, -0.403736793498034, 0.231496128244633, -0.422372407926746, 0.374118394695298, -0.366005774997015, 1.19010144652106, -0.737327525257229, 0.290666645439266, -0.884849568284455, 0.208006478870832, -0.0477301724785263, -1.68452064613859, -0.144226556618951, 1.18021366550265, 0.681399923199504, 0.143247630887551, -1.19231644371292, 1.16922865278909, 0.0792017089453723, -0.451773752768677, 1.64202821280077, -0.769592321599518, 0.303360960757848, 1.28173742118351, 0.602222795170429, -0.307022264536836, -0.418418103422641, 0.355135530046085, 0.513481114599557, 0.01860740032874, 1.31844897225668, -0.065831999838153, -0.700296078317704, 0.537326131521341, -2.20178232235392, 0.391973743798411, 0.496960952423719, -0.224874715429902, -1.11714316533188, -0.394994603313877, 1.54983034223631, -0.743514479798666, -2.33171211772957, 0.812245442209214, -0.501310657201351, -0.510886565952357, -1.21536404115501, -0.022558628347222, 0.701239300429707, -0.587482025589799, -0.60672794141879, 1.09664021500029, -0.247509677080296, -0.159901713323247, -0.625778250735075, 0.900434635600238, -0.994193629266225, 0.849250385880362, 0.805702288976437, -0.46760093599122, 0.848420313723343, 0.986769863596017, 0.57562028851936, 2.02484204541817, -1.96235319122504, -1.16492093063759, -1.37651921373083, 0.167679934469767, 1.58462907915972, 1.67788895297625, 0.488296698394955, 0.878673262560324, -0.144874874029881, 0.468971760074208, 0.376235477129484, -0.761040275056481, -0.293294933750864, -0.134841264407518, 1.39384581617302, -1.03698868969829, -2.11433514780366, 0.768278218204398, -0.816160620753485, -0.436106923160574, 0.904705031282809, -0.763086264548317, -0.341066979637804, 1.50242453423459, 0.5283077123164, 0.542191355464308, -0.136673355717877, -1.13673385330273, -1.49662715435579, -0.223385643556595, 2.00171922777288, 0.221703816213622, 0.164372909246388, 0.33262360885001, -0.385207999091431, -1.39875402655789, 2.67574079585182, -0.423686088681983, -0.298601511933064, -1.79234172685298, -0.248008225068098, -0.247303918374605, -0.255510378526346, -1.78693810009257, 1.78466281602476, 1.76358634774588, 0.689600221934096, -1.10074064442392, 0.714509356897811, -0.246470316934021, -0.319786165927205, 1.3626442929547, -1.22788258998268, -0.511219232750473, -0.731194999064964, 0.0197520068767907, -1.57286391470999, -0.703333269828288, 0.715932088907665, 0.465214906423109, -0.973902306404292, 0.559217730468333, -2.43263974540567, -0.340484926702145, 0.713033194850233, -0.659037386489128, -0.0364026225664853, -1.59328630152691, 0.847792797247769, -1.85038884866061, -0.323650631690314, -0.255248112503172, 0.0609212273067499, -0.823491629018878, 1.82973048483604, -1.4299162157815, 0.254137143004585, -2.93977369533598, 0.00241580894347513, 0.509665571261936, -1.08472000111396, 0.704832976877566, 0.330976349994212, 0.976327472976372, -0.843339880398736, -0.970579904862913, -1.77153134864293, -0.322470341645949, -1.3388007423545, 0.688156028146055, 0.0712806522505788, 2.18975235927918, -1.15770759930148, 1.18168806409233, -0.527368361608178, -1.45662801131076, 0.572967370493168, -1.43337770467221, -1.0551850185265, -0.733111877421952, 0.210907264251298, -0.99892072706101, 1.07785032320562, -1.19897438250787, 0.216637035184711, 0.143087029788927, -1.06575009105298, -0.428623410701779, -0.656179476771346, 0.959394326857613, 1.55605263610222, -1.04079643378825, 0.930572408540247, -0.0754459310121911, -1.96719534905096, -0.755903642645193, 0.461149160506762, 0.145106631047872, -2.44231132108142, 0.580318685451573, 0.655051998486745, -0.304508837259419, -0.707568232617297, 1.97157201447823, -0.0899986806477491, -0.0140172519319242, -1.12345693711015, -1.34413012337621, -1.52315577106209, -0.421968210382174, 1.36092446408098, 1.75379484533487, 1.56836473435279, 1.29675556994775, -0.237596251443612, -1.22415013568071, -0.327812680140072, -2.41245027627225, -0.31379287344339, 1.65987870464572, 0.130953100607517, 1.09588868268958, 0.489340956943913, -0.778910295448653, 1.74355935292935, -0.0783872853310662, -0.975553792522548, 0.0706598247385691, -1.5185995288232, 0.863779032791318, 0.501568385364368, -0.354781329646987, -0.48842888759119, 0.936293947292698, -1.06240838642214, -0.983820871704388, 0.424247877020078, -0.451313480552634, 0.925084796723901, -0.198620809812313, 1.1948510169004, 0.495544705180151, -2.24515257409297, -1.33537136041822, 1.28277520617769, 0.690795904173712, -0.967062667858309, -1.34579368502011, 1.03366538931573, -0.811776459663054, 1.80172547861817, 1.77154196037941, -1.45469136980239, -0.845654312575637, -1.25047965932418, 0.667288067904176, -1.29076968763763, -2.03500353846891, 2.02134699449252, 1.00597348761217, 0.817123601225002, -0.663988283577244, -0.0112812312503971, 0.619677256209794, -1.2812387419642, -0.124261326376048, 0.175741654501771, 1.69277379447271, 0.642132717206933, 1.28223303234921, 0.140546973547383, -1.11250268318057, -0.339676971961898, -1.66476463926247, 0.928851761720271, 1.41682683471578, -0.0627207776270142, -0.980902336097871, 1.08715025123701, 0.139327026560686, -0.386272097673785, 1.12358542318002, -0.759845657357593, 1.14895910322903, -0.84247625893675, 0.391413340392769, 0.891377242318638, -1.33525871255432, 0.398123477834933, -0.111586802948826, 0.675743916642286, -0.788597935379218, -0.0869863340574937, 1.38228400775579, 0.168490163808026, 0.823190948275118, -0.220894597616442, -1.02939165471471, -0.0109256908471287, -1.22499115544077, -2.59611138844997, 1.16912259199149, -1.08690881697743, -1.82608301264641, 0.995281807292567, -0.0118617814307389, -0.599628394751609, -0.177947986602105, -0.425981341802302, 0.996658776455071, 0.727660708501531, -1.72663059586351, 0.35339849564175, 0.726813665899557, 0.668260975976705, -2.42431730928421, -0.235357425015276, 1.97963332091796, 0.796794538639531, -1.70927618064586, -1.66366871188344, 0.49110955231555, -0.174055485572285, 0.961290563877483, 0.293826661677307, 0.0809993635091707, 0.183661842799281, 0.16625503539133, -1.26959906621425, 2.34949332061946, -1.41200540742533, -0.0169614927822389, -0.544319352621665, 1.80011233335541, 1.01144017617923, -0.563716555906629, 0.205420795375023, 1.16546195019875, 2.2363228395069, 0.302265076168852, -1.04250660223103, -0.983542313495811, 2.00571858043419, -2.07057148385748, 3.05574236888186, -0.2613505940873, -0.45439325911372, 0.157560555463951, 0.933388727865257, 0.302828275834352, -1.95615022247248, 0.353536709469366, 0.450424513742423, 0.659550870534498, -1.03142072760739, -2.37102288136148, -0.324576307824521, -0.94429875090974, -0.765889998443101, -0.953779266719918, -0.398004449854522, -0.311217062923606, 0.796092713239834, 0.986428343875779, -0.794531662417426, -0.308817971515212, 0.361444766025588, 1.39879110483802, -0.0560704196424014, -1.69887349155411, 0.231852545141301, -0.119090670970556, 1.77249285346043, 0.343422165443603, -0.623049782354728, -0.439522293828527, -0.505296789545481, 0.186035137157474, 0.176417797599477, 0.915848206824781, 0.320176726180939, -0.36668729712132, -0.940611295179737, 0.634702931419534, -0.0624884823283309, 0.182837867691791, 1.10364101842281, 1.75203561864876, -0.953816460135553, 1.64408045682029, -0.866733529450919, 0.266352192194765, 0.222370494855364, -0.276908509892329, 1.39425304101804, -0.658911999778006, 0.660531280513291, -0.0132557001432984, -0.931480298730843, 1.21468914346638, -2.08834036550883, -0.526152469412621, -1.54140256690726, 0.194321069901269, 0.264422549440984, -1.11873516870975, 0.650952956147074, -1.032900238582, 0.659201499417982, 0.237829399627714, 0.71527594710452, -0.93848303919385, 0.0953540096781045, -0.462819419956343, -1.46888215609457, 0.152686505521528, 1.77376261131718, -0.648070933515048, -0.199817475668204, 0.689243732977183, 0.0361455098365949, 1.94353631225286, 0.737213733789873, 2.32133393302204, 0.34890934629397, -1.13391666127592, 0.421335267904323, -0.924556256814186, -1.00706236611934, -0.189474330103611, 0.9339167001786, 0.343909999577983, 0.814020276463554, 0.915341002280603, -0.171852130422256, -2.40223111360906, 0.795906851730836, 2.16911586935979, 0.0583834994596872, -1.35491444952768, -0.36755076555405, -0.934517579155246, -0.0416392200716165, 0.676112005556641, 0.86643614680554, 0.235175024813313, -0.933970133276432, 0.813252167281602, 1.34831855513005, 2.25188277593107, -0.493652434465923, 0.474092599152496, 1.19369166611409, -0.116395471522116, 0.524858555768479, 0.214421481278423, -0.134449534536861, 0.168509102875034, 0.964732769529999, 0.408778914004396, -0.466438215174507, -2.23978297582177, -0.795063185124836, -0.0199551173436351, -2.51442512196431, 2.21095203169735, -1.48876222991277, -1.16075187561923, 1.45773822495573, -2.18987594871999, 0.739062681128372, -0.344008768542759, 0.455876741511694, 0.878110262544797, -0.958463922912324, -0.70584484574861, -2.99694930322778, -0.961052324044844, 0.380188253916704, 0.505067589582035, 2.02705600379976, 0.0646270287396298, 0.463658648401867, 0.0747783326935444, -0.48683623583544, 0.748910820862606, 0.464234580889638, 0.129420458393473, -0.815460651610283, -0.0401504517354262, 0.781416977680249, 0.676452747462629, -0.482652525567947, -0.669113467963509, 0.512801321922257, 1.04890992895349, 0.121058163102937, -0.313292885379532, -0.88067073001145, -0.419286891272189, -1.4827516780218, -0.697318199400121;
            blah.transposeInPlace();
            return blah;
          };
*/
          auto Ykt = Ykt_diff.col(iter);
          auto Skt = Skt_diff.col(iter);
          auto alpha = alpha_mat.col(iter);
          std::vector<size_t> ys_cols;
          const size_t curr_hist_size3
              = iter < history_size ? iter + 1 : history_size;
          {
            for (Eigen::Index end_iter = iter; end_iter >= 0; --end_iter) {
              if (check_curvatures_vec[end_iter]) {
                ys_cols.push_back(end_iter);
              }
              if (ys_cols.size() == history_size) {
                break;
              }
            }
          }
          std::vector<decltype(Ykt_diff.col(0))> Ykt_h;
          std::vector<decltype(Skt_diff.col(0))> Skt_h;
          std::for_each(
              ys_cols.rbegin(), ys_cols.rend(),
              [&Ykt_h, &Skt_h, &Ykt_diff, &Skt_diff](const size_t idx) {
                Ykt_h.push_back(Ykt_diff.col(idx));
                Skt_h.push_back(Skt_diff.col(idx));
              });
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
           * 3a. Run BFGS-Sample to get `num_elbo_draws` draws from normal
           * approximation and log density of draws in the approximate normal
           * distribution
           */
          taylor_approx_t taylor_appx_tuple = construct_taylor_approximation(
              Ykt_h, alpha, Dk, ninvRST, param_mat.col(iter + 1),
              grad_mat.col(iter + 1));

          auto elbo = est_elbo_draws(taylor_appx_tuple, num_elbo_draws, alpha,
                                     fn, rnorm, model, iter);
          // TODO: Calculate total function calls
          // fn_call = fn_call + DIV_fit$fn_calls_DIV
          // DIV_ls = c(DIV_ls, DIV_fit$elbo)
          //  4. Find $l \in L$ that maximizes ELBO $l^* = arg max_l ELBO^(l)$.
          if (refresh > 0
              && (lbfgs.iter_num() == 0 || ((lbfgs.iter_num() + 1) % refresh == 0))) {
                logger.info(iter_msg + ": ELBO (" + std::to_string(elbo) + ")");
              }
          {
            std::lock_guard<std::mutex> guard(update_best_mutex);
            if (STAN_DEBUG_PATH_BEST_ELBO) {
              // std::cout << "elbo curr: " << DIV_fit.elbo << "\n";
              std::cout << "elbo best: " << elbo_max << "\n";
            }
            if (elbo > elbo_max) {
              elbo_max = elbo;
              taylor_approx_best = std::move(taylor_appx_tuple);
              best_E = iter;
            }
          }
        }
      });
  // std::cout << "Winner: " << best_E << "\n";
  boost::variate_generator<boost::ecuyer1988&, boost::normal_distribution<>>
      rand_unit_gaus(rng, boost::normal_distribution<>());
  auto rnorm
      = [&rand_unit_gaus, num_params = param_size, num_samples = num_draws]() {
          return Eigen::MatrixXd::NullaryExpr(
              num_params, num_samples,
              [&rand_unit_gaus]() { return rand_unit_gaus(); });
        };
  auto draws_tuple = approximation_samples(taylor_approx_best, num_draws,
                                           alpha_mat.col(best_E), rnorm);
  auto&& draws_mat = std::get<0>(draws_tuple);
  auto&& lp_approx_vec = std::get<1>(draws_tuple);
  Eigen::MatrixXd constrainted_draws_mat(names.size(), draws_mat.cols());
  Eigen::VectorXd lp_ratio(draws_mat.cols());
  tbb::parallel_for(
      tbb::blocked_range<Eigen::Index>(0, draws_mat.cols()),
      [&](tbb::blocked_range<Eigen::Index> r) {
        Eigen::VectorXd unconstrained_draws;
        Eigen::VectorXd constrained_draws1;
        Eigen::VectorXd constrained_draws2(names.size());
        for (int i = r.begin(); i < r.end(); ++i) {
          unconstrained_draws = draws_mat.col(i);
          model.write_array(rng, unconstrained_draws, constrained_draws1);
          //constrainted_draws_mat.col(i) = constrained_draws1;
          constrained_draws2.head(names.size() - 2) = constrained_draws1;
          constrained_draws2(names.size() - 2) = lp_approx_vec(i);
          constrained_draws2(names.size() - 1) = -fn(unconstrained_draws);
          lp_ratio(i) = - constrained_draws2(names.size() - 1) - constrained_draws2(names.size() - 2);
          constrainted_draws_mat.col(i) = constrained_draws2;
        }
      });
      parameter_writer(constrainted_draws_mat);
  return ret_pathfinder<ReturnLpSamples>(0, std::move(lp_ratio), std::move(constrainted_draws_mat));
}

}  // namespace optimize
}  // namespace services
}  // namespace stan
#endif
