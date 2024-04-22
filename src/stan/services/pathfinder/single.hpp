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
#include <stan/services/util/duration_diff.hpp>
#include <boost/circular_buffer.hpp>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_queue.h>
#include <tbb/task_group.h>
#include <string>
#include <vector>
#include <atomic>

namespace stan {
namespace services {
namespace pathfinder {
namespace internal {

/**
 * Check the optimization direction is strictly positive and curvature is 'tame'
 * @tparam EigVec1 Type derived from `Eigen::DenseBase` with one column at
 * compile time
 * @param Yk Vector of gradients
 * @param Sk Vector of values
 * @return boolean with true if both the optimization direction `Dk` is greater
 * than zero and the curvature `thetak` is less than 1e12.
 */
template <typename EigVec, stan::require_eigen_vector_t<EigVec>* = nullptr>
inline bool check_curve(const EigVec& Yk, const EigVec& Sk) {
  auto Dk = Yk.dot(Sk);
  auto thetak = std::abs(Yk.array().square().sum() / Dk);
  return Dk > 0 && thetak <= 1e12;
}

/**
 * eq 4.9
 * Gilbert, J.C., Lemaréchal, C. Some numerical experiments with
 * variable-storage quasi-Newton algorithms. Mathematical Programming 45,
 * 407–435 (1989). https://doi.org/10.1007/BF01589113
 * @tparam EigVec1 Type derived from `Eigen::DenseBase` with one column at
 * compile time
 * @tparam EigVec2 Type derived from `Eigen::DenseBase` with one column at
 * compile time
 * @tparam EigVec3 Type derived from `Eigen::DenseBase` with one column at
 * compile time
 * @param alpha_init Vector of initial values to update
 * @param Yk Vector of gradients
 * @param Sk Vector of values
 * @return A vector of the next updated diagonal of the hessian.
 */
template <typename EigVec1, typename EigVec2, typename EigVec3>
inline auto form_diag(const EigVec1& alpha_init, const EigVec2& Yk,
                      const EigVec3& Sk) {
  double y_alpha_y = Yk.dot(alpha_init.asDiagonal() * Yk);
  double y_s = Yk.dot(Sk);
  double s_inv_alpha_s
      = Sk.dot(alpha_init.array().inverse().matrix().asDiagonal() * Sk);
  return y_s
         / (y_alpha_y / alpha_init.array() + Yk.array().square()
            - (y_alpha_y / s_inv_alpha_s)
                  * (Sk.array() / alpha_init.array()).square());
}

/**
 * Information from running the taylor approximation
 */
struct taylor_approx_t {
  Eigen::VectorXd x_center;  // Mean estimate
  double logdetcholHk;       // Log deteriminant of the cholesky
  Eigen::MatrixXd L_approx;  // Approximate choleskly
  Eigen::MatrixXd Qk;  // Q of the QR decomposition. Only used for sparse approx
  Eigen::VectorXd alpha;  // diagonal of the initial inv hessian
  bool use_full;  // boolean indicating if full or sparse approx was used.
};

/**
 * Information from calling ELBO estimation
 */
struct elbo_est_t {
  // Evidence Lower Bound
  double elbo{-std::numeric_limits<double>::infinity()};
  size_t fn_calls{0};  // Number of times the log_prob function is called.
  Eigen::MatrixXd repeat_draws;  // Samples
  // Two column matrix. First column is approximate lp and second is true lp
  Eigen::Array<double, Eigen::Dynamic, 2> lp_mat;
  // Ratio of approximate lp to true lp.
  Eigen::Array<double, Eigen::Dynamic, 1> lp_ratio;
};

/**
 * Generate approximate draws using either the full or sparse taylor
 * approximation.
 * @tparam EigMat A type inheriting from `Eigen::DenseBase` with dynamic rows
 * and columns.
 * @tparam EigVec A type inheriting from `Eigen::DenseBase` with the compile
 * time number of columns equal to 1.
 * @param u A matrix of gaussian IID samples with rows equal to the size of the
 * number of samples to be made and columns equal to the number of parameters.
 * @param taylor_approx Approximation from `taylor_approximation`.
 * @return A matrix with rows equal to the number of samples and columns equal
 * to the number of parameters.
 */
template <typename EigMat, require_eigen_matrix_dynamic_t<EigMat>* = nullptr>
inline Eigen::MatrixXd approximate_samples(
    EigMat&& u, const taylor_approx_t& taylor_approx) {
  if (taylor_approx.use_full) {
    return (taylor_approx.L_approx.transpose() * std::forward<EigMat>(u))
               .colwise()
           + taylor_approx.x_center;
  } else {
    return (taylor_approx.alpha.array().sqrt().matrix().asDiagonal()
            * (taylor_approx.Qk
                   * (taylor_approx.L_approx
                      - Eigen::MatrixXd::Identity(
                          taylor_approx.L_approx.rows(),
                          taylor_approx.L_approx.cols()))
                   * (taylor_approx.Qk.transpose() * u)
               + u))
               .colwise()
           + taylor_approx.x_center;
  }
}

/**
 * Generate approximate draws using either the full or sparse taylor
 * approximation.
 * @tparam EigVec1 A type inheriting from `Eigen::DenseBase` with the compile
 * time number of columns equal to 1.
 * @tparam EigVec2 A type inheriting from `Eigen::DenseBase` with the compile
 * time number of columns equal to 1.
 * @param u A matrix of gaussian IID samples with columns equal to the size of
 * the number of samples to be made and rows equal to the number of parameters.
 * @param taylor_approx Approximation from `taylor_approximation`.
 * @return A matrix with columns equal to the number of samples and rows equal
 * to the number of parameters. Each column represents an approximate draw for
 * the set of parameters.
 * @return A vector of an approximated sample derived from the taylor
 * approximation.
 */
template <typename EigVec1, typename EigVec2,
          require_eigen_vector_t<EigVec1>* = nullptr>
inline Eigen::VectorXd approximate_samples(
    EigVec1&& u, const taylor_approx_t& taylor_approx) {
  if (taylor_approx.use_full) {
    return (taylor_approx.L_approx.transpose() * std::forward<EigVec1>(u))
           + taylor_approx.x_center;
  } else {
    return (taylor_approx.alpha.array().sqrt().matrix().asDiagonal()
            * (taylor_approx.Qk
                   * (taylor_approx.L_approx
                      - Eigen::MatrixXd::Identity(
                          taylor_approx.L_approx.rows(),
                          taylor_approx.L_approx.cols()))
                   * (taylor_approx.Qk.transpose() * u)
               + u))
           + taylor_approx.x_center;
  }
}

/**
 * Generate an Eigen matrix of from an rng generator.
 * @tparam RowsAtCompileTime The number of compile time rows for the result
 * matrix.
 * @tparam ColsAtCompileTime The number of compile time cols for the result
 * matrix.
 * @tparam Generator A functor with a valid `operator()` used to generate the
 * samples.
 * @param[in,out] variate_generator An rng generator
 * @param num_params The runtime number of parameters
 * @param num_samples The runtime number of samples.
 * @return A matrix of values generated from the `variate_generator`
 */
template <Eigen::Index RowsAtCompileTime = Eigen::Dynamic,
          Eigen::Index ColsAtCompileTime = Eigen::Dynamic, typename Generator>
inline Eigen::Matrix<double, RowsAtCompileTime, ColsAtCompileTime>
generate_matrix(Generator&& variate_generator, const Eigen::Index num_params,
                const Eigen::Index num_samples) {
  return Eigen::Matrix<double, RowsAtCompileTime, ColsAtCompileTime>::
      NullaryExpr(num_params, num_samples,
                  [&variate_generator]() { return variate_generator(); });
}

/**
 * Estimate the approximate draws given the taylor approximation.
 *
 * @tparam ReturnElbo If true, calculate ELBO and return it in `elbo_est_t`. If
 * `false` ELBO is set in the return as `-Infinity`
 * @tparam LPF Type of log probability functor
 * @tparam ConstrainF Type of functor for constraining parameters
 * @tparam RNG Type of random number generator
 * @tparam EigVec Type inheriting from `Eigen::DenseBase` with 1 column at
 * compile time.
 * @tparam Logger Type of logger callback
 * @param lp_fun Functor to calculate the log density
 * @param constrain_fun A functor to transform parameters to the constrained
 * space
 * @param[in,out] rng A generator to produce standard gaussian random variables
 * @param taylor_approx The taylor approximation at this iteration of LBFGS
 * @param num_samples Number of approximate samples to generate
 * @param alpha The approximation of the diagonal hessian
 * @param iter_msg The beginning of messages that includes the iteration number
 * @param logger A callback writer for messages
 * @param calculate_lp If true, calculate the log probability of the samples.
 * Else set to `NaN` for each sample.
 * @return A struct with the ELBO estimate along with the samples and log
 * probability ratios.
 */
template <bool ReturnElbo = true, typename LPF, typename ConstrainF,
          typename RNG, typename EigVec, typename Logger>
inline elbo_est_t est_approx_draws(LPF&& lp_fun, ConstrainF&& constrain_fun,
                                   RNG&& rng,
                                   const taylor_approx_t& taylor_approx,
                                   size_t num_samples, const EigVec& alpha,
                                   const std::string& iter_msg, Logger&& logger,
                                   bool calculate_lp = true) {
  boost::variate_generator<stan::rng_t&, boost::normal_distribution<>>
      rand_unit_gaus(rng, boost::normal_distribution<>());
  const auto num_params = taylor_approx.x_center.size();
  size_t lp_fun_calls = 0;
  Eigen::MatrixXd unit_samps
      = generate_matrix(rand_unit_gaus, num_params, num_samples);
  Eigen::Array<double, Eigen::Dynamic, 2> lp_mat(num_samples, 2);
  lp_mat.col(0) = (-taylor_approx.logdetcholHk)
                  + -0.5
                        * (unit_samps.array().square().colwise().sum()
                           + num_params * stan::math::LOG_TWO_PI);
  Eigen::MatrixXd approx_samples
      = approximate_samples(std::move(unit_samps), taylor_approx);
  const auto log_stream
      = [](auto& logger, auto& pathfinder_ss, const auto& iter_msg) {
          if (pathfinder_ss.str().length() == 0)
            return;
          logger.info(iter_msg + pathfinder_ss.str());
          pathfinder_ss.str(std::string());
        };
  Eigen::VectorXd approx_samples_col;
  std::stringstream pathfinder_ss;
  Eigen::Array<double, Eigen::Dynamic, 1> lp_ratio;
  if (calculate_lp) {
    for (Eigen::Index i = 0; i < num_samples; ++i) {
      try {
        approx_samples_col = approx_samples.col(i);
        ++lp_fun_calls;
        lp_mat.coeffRef(i, 1) = lp_fun(approx_samples_col, pathfinder_ss);
      } catch (const std::domain_error& e) {
        lp_mat.coeffRef(i, 1) = -std::numeric_limits<double>::infinity();
      }
      log_stream(logger, pathfinder_ss, iter_msg);
    }
    lp_ratio = lp_mat.col(1) - lp_mat.col(0);
  } else {
    lp_ratio = Eigen::Array<double, Eigen::Dynamic, 1>::Constant(
        lp_mat.rows(), std::numeric_limits<double>::quiet_NaN());
    lp_mat.col(1) = Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(
        lp_mat.rows(), std::numeric_limits<double>::quiet_NaN());
  }
  if (ReturnElbo) {
    double elbo = lp_ratio.mean();
    return elbo_est_t{elbo, lp_fun_calls, std::move(approx_samples),
                      std::move(lp_mat), std::move(lp_ratio)};
  } else {
    return elbo_est_t{-std::numeric_limits<double>::infinity(), lp_fun_calls,
                      std::move(approx_samples), std::move(lp_mat),
                      std::move(lp_ratio)};
  }
}

/**
 * Construct the full taylor approximation
 * @tparam GradMat Type inheriting from `Eigen::DenseBase` with compile time
 * dynamic rows and columns
 * @tparam AlphaVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam DkVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam InvMat Type inheriting from `Eigen::DenseBase` with dynamic compile
 * time rows and columns
 * @tparam EigVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @param Ykt_mat Matrix of the changes to the gradient with column length of
 * history size.
 * @param alpha The diagonal of the approximate hessian
 * @param Dk vector of Columnwise products of parameter and gradients with size
 * equal to history size
 * @param ninvRST Inverse of the Rk matrix
 * @param point_est The parameters for the given iteration of LBFGS
 * @param grad_est The gradients for the given iteration of LBFGS
 * @return The components of the dense taylor approximation
 */
template <typename GradMat, typename AlphaVec, typename DkVec, typename InvMat,
          typename EigVec>
inline taylor_approx_t taylor_approximation_dense(
    GradMat&& Ykt_mat, const AlphaVec& alpha, const DkVec& Dk,
    const InvMat& ninvRST, const EigVec& point_est, const EigVec& grad_est) {
  auto y_tcrossprod_alpha_expr
      = Ykt_mat.transpose() * alpha.array().sqrt().matrix().asDiagonal();
  Eigen::MatrixXd y_tcrossprod_alpha
      = Eigen::MatrixXd(y_tcrossprod_alpha_expr.rows(),
                        y_tcrossprod_alpha_expr.rows())
            .setZero()
            .selfadjointView<Eigen::Lower>()
            .rankUpdate(std::move(y_tcrossprod_alpha_expr));
  /*
   * + DK.asDiagonal() cannot be done on same line
   * See https://forum.kde.org/viewtopic.php?f=74&t=136617
   */
  y_tcrossprod_alpha += Dk.asDiagonal();

  Eigen::MatrixXd y_mul_alpha = Ykt_mat.transpose() * alpha.asDiagonal();
  Eigen::MatrixXd Hk
      = y_mul_alpha.transpose() * ninvRST
        + ninvRST.transpose() * (y_mul_alpha + y_tcrossprod_alpha * ninvRST);
  Hk += alpha.asDiagonal();
  Eigen::MatrixXd L_hk = Hk.llt().matrixL().transpose();
  double logdetcholHk = L_hk.diagonal().array().abs().log().sum();
  Eigen::VectorXd x_center = point_est - Hk * grad_est;
  return {std::move(x_center),   logdetcholHk, std::move(L_hk),
          Eigen::MatrixXd(0, 0), alpha,        true};
}

/**
 * Construct the sparse taylor approximation
 * @tparam GradMat Type inheriting from `Eigen::DenseBase` with compile time
 * dynamic rows and columns
 * @tparam AlphaVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam DkVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam InvMat Type inheriting from `Eigen::DenseBase` with dynamic compile
 * time rows and columns
 * @tparam EigVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @param Ykt_mat Matrix of the changes to the gradient with column length of
 * history size.
 * @param alpha The diagonal of the approximate hessian
 * @param Dk vector of Columnwise products of parameter and gradients with size
 * equal to history size
 * @param ninvRST The solution of X = R^-1 * S
 * @param point_est The parameters for the given iteration of LBFGS
 * @param grad_est The gradients for the given iteration of LBFGS
 * @return The components of the sparse taylor approximation
 */
template <typename GradMat, typename AlphaVec, typename DkVec, typename InvMat,
          typename EigVec>
inline taylor_approx_t taylor_approximation_sparse(
    GradMat&& Ykt_mat, const AlphaVec& alpha, const DkVec& Dk,
    const InvMat& ninvRST, const EigVec& point_est, const EigVec& grad_est) {
  const Eigen::Index history_size = Ykt_mat.cols();
  const Eigen::Index history_size_times_2 = history_size * 2;
  const Eigen::Index num_params = alpha.size();
  Eigen::MatrixXd y_mul_sqrt_alpha
      = Ykt_mat.transpose() * alpha.array().sqrt().matrix().asDiagonal();
  Eigen::MatrixXd Wkbart(history_size_times_2, num_params);
  Wkbart.topRows(history_size) = y_mul_sqrt_alpha;
  Wkbart.bottomRows(history_size)
      = ninvRST * alpha.array().inverse().sqrt().matrix().asDiagonal();
  Eigen::MatrixXd Mkbar(history_size_times_2, history_size_times_2);
  Mkbar.topLeftCorner(history_size, history_size).setZero();
  Mkbar.topRightCorner(history_size, history_size)
      = Eigen::MatrixXd::Identity(history_size, history_size);
  Mkbar.bottomLeftCorner(history_size, history_size)
      = Eigen::MatrixXd::Identity(history_size, history_size);
  Eigen::MatrixXd y_tcrossprod_alpha
      = Eigen::MatrixXd(y_mul_sqrt_alpha.rows(), y_mul_sqrt_alpha.rows())
            .setZero()
            .selfadjointView<Eigen::Lower>()
            .rankUpdate(std::move(y_mul_sqrt_alpha));
  y_tcrossprod_alpha += Dk.asDiagonal();
  Mkbar.bottomRightCorner(history_size, history_size) = y_tcrossprod_alpha;
  Wkbart.transposeInPlace();
  const auto min_size = std::min(num_params, history_size_times_2);
  // Note: This is doing the QR decomp inplace using Wkbart's memory
  Eigen::HouseholderQR<Eigen::Ref<decltype(Wkbart)>> qr(Wkbart);
  Eigen::MatrixXd Rkbar
      = qr.matrixQR().topLeftCorner(min_size, history_size_times_2);
  Rkbar.triangularView<Eigen::StrictlyLower>().setZero();
  Eigen::MatrixXd Qk
      = qr.householderQ() * Eigen::MatrixXd::Identity(num_params, min_size);
  Eigen::MatrixXd L_approx
      = (Rkbar.triangularView<Eigen::Upper>() * Mkbar
             * Rkbar.transpose().triangularView<Eigen::Lower>()
         + Eigen::MatrixXd::Identity(min_size, min_size))
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
           + (alpha.array() * (Ykt_mat * ninvRSTg).array()).matrix()
           + (ninvRST.transpose()
              * ((Ykt_mat.transpose() * alpha_mul_grad)
                 + y_tcrossprod_alpha * ninvRSTg)));

  return {std::move(x_center), logdetcholHk, std::move(L_approx),
          std::move(Qk),       alpha,        false};
}

/**
 * Construct the taylor approximation.
 * @tparam GradMat Type inheriting from `Eigen::DenseBase` with compile time
 * dynamic rows and columns
 * @tparam AlphaVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam DkVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam InvMat Type inheriting from `Eigen::DenseBase` with dynamic compile
 * time rows and columns
 * @tparam EigVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @param Ykt_mat Matrix of the changes to the gradient with column length of
 * history size.
 * @param alpha The diagonal of the approximate hessian
 * @param Dk vector of Columnwise products of parameter and gradients with size
 * equal to history size
 * @param ninvRST
 * @param point_est The parameters for the given iteration of LBFGS
 * @param grad_est The gradients for the given iteration of LBFGS
 * @return The components of either the sparse or dense taylor approximation
 */
template <typename GradMat, typename AlphaVec, typename DkVec, typename InvMat,
          typename EigVec>
inline taylor_approx_t taylor_approximation(
    GradMat&& Ykt_mat, const AlphaVec& alpha, const DkVec& Dk,
    const InvMat& ninvRST, const EigVec& point_est, const EigVec& grad_est) {
  // If twice the current history size is larger than the number of params
  // use a sparse approximation
  const auto history_size = Ykt_mat.cols();
  const auto num_params = Ykt_mat.rows();
  if (2 * history_size >= num_params) {
    return taylor_approximation_dense(Ykt_mat, alpha, Dk, ninvRST, point_est,
                                      grad_est);
  } else {
    return taylor_approximation_sparse(Ykt_mat, alpha, Dk, ninvRST, point_est,
                                       grad_est);
  }
}

/**
 * Construct the return for directly calling single pathfinder or
 * calling single pathfinder from multi pathfinder.
 * @tparam ReturnLpSamples if `true` then this function returns the lp_ratio
 * and samples. If false then only the return code is returned
 * @tparam EigMat A type inheriting from `Eigen::DenseBase`
 * @tparam EigVec A type inheriting from `Eigen::DenseBase` with one column
 * defined at compile time
 * @return A tuple with an error code, a vector holding the log prob ratios,
 * matrix of samples, and an unsigned integer for number of times the log prob
 * functions was called
 */
template <bool ReturnLpSamples, typename EigMat, typename EigVec,
          std::enable_if_t<ReturnLpSamples>* = nullptr>
inline auto ret_pathfinder(int return_code, EigVec&& lp_ratio, EigMat&& samples,
                           const std::atomic<size_t>& lp_calls) {
  return std::make_tuple(return_code, std::forward<EigVec>(lp_ratio),
                         std::forward<EigMat>(samples), lp_calls.load());
}

template <bool ReturnLpSamples, typename EigMat, typename EigVec,
          std::enable_if_t<!ReturnLpSamples>* = nullptr>
inline auto ret_pathfinder(int return_code, EigVec&& lp_ratio, EigMat&& samples,
                           const std::atomic<size_t>& lp_calls) noexcept {
  return return_code;
}

/**
 * Estimate the approximate draws given the taylor approximation.
 * @tparam RNG Type of random number generator
 * @tparam LPFun Type of log probability functor
 * @tparam ConstrainFun Type of functor for constraining parameters
 * @tparam Logger Type inheriting from `stan::callbacks::logger`
 * @tparam AlphaVec Type inheriting from `Eigen::DenseBase` with 1 column at
 * compile time
 * @tparam GradBuffer Boost circular buffer with inner Eigen vector type
 * @tparam CurrentParams Type inheriting from `Eigen::DenseBase` with 1 column
 * at compile time
 * @tparam CurentGrads Type inheriting from `Eigen::DenseBase` with 1 column at
 * compile time
 * @tparam ParamMat Type inheriting from `Eigen::DenseBase` with dynamic rows
 * and columns at compile time.
 * @tparam Logger Type of logger callback
 * @param[in,out] rng A generator to produce standard gaussian random variables
 * @param alpha The approximation of the diagonal hessian
 * @param lp_fun Functor to calculate the log density
 * @param constrain_fun A functor to transform parameters to the constrained
 * space
 * @param current_params Parameters from iteration of LBFGS
 * @param current_grads Gradients from iteration of LBFGS
 * @param Ykt_mat Matrix of the last `history_size` changes in the gradient.
 * @param[in,out] Skt_mat Matrix of the last `history_size` changes in the
 * parameters. `Skt_mat` is transformed in this function and will hold inverse
 * solution of RS^T
 * @param num_elbo_draws Number of draws for the ELBO estimation
 * @param iter_msg The beginning of messages that includes the iteration number
 * @param logger A callback writer for messages
 * @return A pair holding the elbo estimate information and the taylor
 * approximation information.
 */
template <typename RNG, typename LPFun, typename ConstrainFun,
          typename AlphaVec, typename CurrentParams, typename CurrentGrads,
          typename GradMat, typename ParamMat, typename Logger>
auto pathfinder_impl(RNG&& rng, LPFun&& lp_fun, ConstrainFun&& constrain_fun,
                     AlphaVec&& alpha, CurrentParams&& current_params,
                     CurrentGrads&& current_grads, GradMat&& Ykt_mat,
                     ParamMat&& Skt_mat, std::size_t num_elbo_draws,
                     const std::string& iter_msg, Logger&& logger) {
  const auto history_size = Ykt_mat.cols();
  Eigen::MatrixXd Rk = Eigen::MatrixXd::Zero(history_size, history_size);
  Rk.template triangularView<Eigen::Upper>() = Skt_mat.transpose() * Ykt_mat;
  Eigen::VectorXd Dk = Rk.diagonal();
  // Unfolded algorithm in paper for inverse RST
  Rk.triangularView<Eigen::Upper>().solveInPlace(Skt_mat.transpose());
  // Skt_mat is now ninvRST
  Skt_mat = -Skt_mat;
  internal::taylor_approx_t taylor_appx = internal::taylor_approximation(
      Ykt_mat, alpha, Dk, Skt_mat.transpose(), current_params, current_grads);
  try {
    return std::make_pair(internal::est_approx_draws<true>(
                              lp_fun, constrain_fun, rng, taylor_appx,
                              num_elbo_draws, alpha, iter_msg, logger),
                          taylor_appx);
  } catch (const std::domain_error& e) {
    logger.warn(iter_msg + "ELBO estimation failed "
                + " with error: " + e.what());
    return std::make_pair(internal::elbo_est_t{}, internal::taylor_approx_t{});
  }
}
}  // namespace internal

/**
 * Run single path pathfinder with specified initializations and write results
 * to the specified callbacks and it returns a return code.
 * @tparam ReturnLpSamples if `true` single pathfinder returns the lp_ratio
 * vector and approximate samples. If `false` only gives a return code.
 * @tparam Model type of model
 * @tparam DiagnosticWriter Type inheriting from @ref
 * stan::callbacks::structured_writer
 * @tparam ParamWriter Type inheriting from @ref stan::callbacks::writer
 * @param[in] model defining target log density and transforms (log $p$ in
 * paper)
 * @param[in] init ($pi_0$ in paper) var context for initialization. Random
 * initial values will be generated for parameters user has not supplied.
 * @param[in] random_seed seed for the random number generator
 * @param[in] stride_id id to advance the pseudo random number generator
 * @param[in] init_radius A non-negative value to initialize variables uniformly
 * in (-init_radius, init_radius) if not defined in the initialization var
 * context
 * @param[in] max_history_size  Non-negative value for (J in paper) amount of
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
 * @param[in,out] interrupt callback to be called every iteration
 * @param[in] num_elbo_draws (K in paper) number of MC draws to evaluate ELBO
 * @param[in] num_draws (M in paper) number of approximate posterior draws to
 * return
 * @param[in,out] logger Logger for messages
 * @param[in,out] init_writer Writer callback for unconstrained inits
 * @param[in,out] parameter_writer Writer callback for parameter values
 * @param[in,out] diagnostic_writer output for diagnostics values
 * @param[in] calculate_lp Whether single pathfinder should return lp
 * calculations. If `true`, calculates the joint log probability for each
 * sample. If `false`, (`num_draws` - `num_elbo_draws`) of the joint log
 * probability calculations will be `NA` and psis resampling will not be
 * performed. Setting this parameter to `false` will also set all of the lp
 * ratios to `NaN`.
 * @return If `ReturnLpSamples` is `true`, returns a tuple of the error code,
 * approximate draws, and a vector of the lp ratio. If `false`, only returns an
 * error code `error_codes::OK` if successful, `error_codes::SOFTWARE`
 * or `error_codes::CONFIG` for failures
 */
template <bool ReturnLpSamples = false, class Model, typename DiagnosticWriter,
          typename ParamWriter>
inline auto pathfinder_lbfgs_single(
    Model& model, const stan::io::var_context& init, unsigned int random_seed,
    unsigned int stride_id, double init_radius, int max_history_size,
    double init_alpha, double tol_obj, double tol_rel_obj, double tol_grad,
    double tol_rel_grad, double tol_param, int num_iterations,
    int num_elbo_draws, int num_draws, bool save_iterations, int refresh,
    callbacks::interrupt& interrupt, callbacks::logger& logger,
    callbacks::writer& init_writer, ParamWriter& parameter_writer,
    DiagnosticWriter& diagnostic_writer, bool calculate_lp = true) {
  const auto start_pathfinder_time = std::chrono::steady_clock::now();
  stan::rng_t rng = util::create_rng(random_seed, stride_id);
  std::vector<int> disc_vector;
  std::vector<double> cont_vector;

  try {
    cont_vector = util::initialize<false>(model, init, rng, init_radius, false,
                                          logger, init_writer);
  } catch (const std::exception& e) {
    logger.error(e.what());
    return internal::ret_pathfinder<ReturnLpSamples>(
        error_codes::SOFTWARE, Eigen::Array<double, Eigen::Dynamic, 1>(0),
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(0, 0), 0);
  }

  const auto num_parameters = cont_vector.size();
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
  lbfgs_update_t lbfgs_update(max_history_size);
  using Optimizer
      = stan::optimization::BFGSLineSearch<Model, lbfgs_update_t, double,
                                           Eigen::Dynamic, true>;
  Optimizer lbfgs(model, cont_vector, disc_vector, std::move(ls_opts),
                  std::move(conv_opts), std::move(lbfgs_update), &lbfgs_ss);
  const std::string path_num("Path [" + std::to_string(stride_id) + "] :");
  if (refresh != 0) {
    logger.info(path_num + "Initial log joint density = "
                + std::to_string(lbfgs.logp()));
  }
  std::vector<std::string> names;
  names.push_back("lp_approx__");
  names.push_back("lp__");
  model.constrained_param_names(names, true, true);
  parameter_writer(names);
  int ret = 0;
  boost::circular_buffer<Eigen::VectorXd> param_buff(max_history_size);
  boost::circular_buffer<Eigen::VectorXd> grad_buff(max_history_size);
  Eigen::VectorXd prev_params
      = Eigen::Map<Eigen::VectorXd>(cont_vector.data(), cont_vector.size());
  std::size_t history_size = 0;
  Eigen::VectorXd prev_grads(num_parameters);
  stan::model::log_prob_grad<true, true>(model, prev_params, prev_grads);
  if (unlikely(save_iterations)) {
    diagnostic_writer.begin_record();
    diagnostic_writer.begin_record("0");
    diagnostic_writer.write("iter", static_cast<int>(0));
    diagnostic_writer.write("unconstrained_parameters", prev_params);
    diagnostic_writer.write("grads", prev_grads);
    diagnostic_writer.end_record();
  }
  auto constrain_fun = [&model](auto&& rng, auto&& unconstrained_draws,
                                auto&& constrained_draws) {
    model.write_array(rng, unconstrained_draws, constrained_draws);
    return constrained_draws;
  };
  auto lp_fun = [&model](auto&& u, auto&& streamer) {
    return model.template log_prob<false, true>(u, &streamer);
  };
  Eigen::VectorXd alpha = Eigen::VectorXd::Ones(num_parameters);
  Eigen::Index best_iteration = -1;
  internal::elbo_est_t elbo_best;
  internal::taylor_approx_t taylor_approx_best;
  std::size_t num_evals{lbfgs.grad_evals()};
  Eigen::MatrixXd Ykt_mat(num_parameters, max_history_size);
  Eigen::MatrixXd Skt_mat(num_parameters, max_history_size);
  std::string log_header = path_num + " Iter      log prob        ||dx||      "
  "||grad||     alpha      alpha0      # evals       ELBO    Best ELBO        "
  "Notes \n";
  auto print_log_remainder = [](auto&& write_log_cond, auto&& msg, auto ret,
                                auto num_evals, auto&& lbfgs, auto best_elbo,
                                auto elbo, auto&& lbfgs_ss,
                                auto&& logger) mutable {
    if (write_log_cond) {
      msg << std::setw(10) << num_evals << std::setw(11) << std::scientific
          << std::setprecision(3) << elbo << std::setw(11) << std::scientific
          << std::setprecision(3) << best_elbo << std::setw(18) << lbfgs.note();
      logger.info(msg.str());
      msg.clear();
      msg.str("");
    }
  };

  while (ret == 0) {
    std::stringstream msg;
    interrupt();
    ret = lbfgs.step();
    double lp = lbfgs.logp();
    bool write_log_cond
        = refresh > 0
          && (ret != 0 || !lbfgs.note().empty() || lbfgs.iter_num() == 0
              || ((lbfgs.iter_num() + 1) % refresh == 0));
    if (write_log_cond) {
      msg << std::setw(5) << log_header << std::setw(15) << lbfgs.iter_num()
          << std::setw(16) << std::scientific << std::setprecision(3) << lp
          << std::setw(15) << std::scientific << std::setprecision(3)
          << lbfgs.prev_step_size() << std::setw(12) << std::scientific
          << std::setprecision(3) << lbfgs.curr_g().norm() << std::setw(13)
          << std::scientific << std::setprecision(3) << lbfgs.alpha()
          << std::setw(11) << std::scientific << std::setprecision(3)
          << lbfgs.alpha0();
    }
    history_size = std::min(history_size + 1,
                            static_cast<std::size_t>(max_history_size));

    if (unlikely(save_iterations)) {
      diagnostic_writer.begin_record(std::to_string(lbfgs.iter_num()));
      diagnostic_writer.write("iter", lbfgs.iter_num());
      diagnostic_writer.write("unconstrained_parameters", prev_params);
      diagnostic_writer.write("grads", prev_grads);
      diagnostic_writer.write("history_size", history_size);
    }
    // if retcode is -1, line search failed w/o updating vals/grads, so exit
    // loop
    if (unlikely(ret == -1)) {
      print_log_remainder(
          write_log_cond, msg, ret, num_evals, lbfgs, elbo_best.elbo,
          std::numeric_limits<double>::quiet_NaN(), lbfgs_ss, logger);
      if (save_iterations) {
        diagnostic_writer.write("lbfgs_success", false);
        diagnostic_writer.write("pathfinder_success", false);
        diagnostic_writer.write("lbfgs_note", lbfgs_ss.str());
        diagnostic_writer.end_record();
      }
      if (lbfgs_ss.str().length() > 0) {
        logger.info(lbfgs_ss);
        lbfgs_ss.str("");
      }
      break;
    }
    try {
      param_buff.push_back(lbfgs.curr_x() - prev_params);
      grad_buff.push_back(lbfgs.curr_g() - prev_grads);
      prev_params = lbfgs.curr_x();
      prev_grads = lbfgs.curr_g();
      auto&& Yk = grad_buff.back();
      auto&& Sk = param_buff.back();
      if (internal::check_curve(Yk, Sk)) {
        double y_alpha_y = Yk.dot(alpha.asDiagonal() * Yk);
        double y_s = Yk.dot(Sk);
        alpha = y_s
                / (y_alpha_y / alpha.array() + Yk.array().square()
                   - (y_alpha_y
                      / Sk.dot(alpha.array().inverse().matrix().asDiagonal()
                               * Sk))
                         * (Sk.array() / alpha.array()).square());
      }
      Eigen::Map<Eigen::MatrixXd> Ykt_map(Ykt_mat.data(), num_parameters,
                                          history_size);
      for (Eigen::Index i = 0; i < history_size; ++i) {
        Ykt_map.col(i) = grad_buff[i];
      }
      Eigen::Map<Eigen::MatrixXd> Skt_map(Skt_mat.data(), num_parameters,
                                          history_size);
      for (Eigen::Index i = 0; i < history_size; ++i) {
        Skt_map.col(i) = param_buff[i];
      }
      std::string iter_msg(path_num + "Iter: ["
                           + std::to_string(lbfgs.iter_num()) + "] ");

      auto pathfinder_res = internal::pathfinder_impl(
          rng, lp_fun, constrain_fun, alpha, lbfgs.curr_x(), lbfgs.curr_g(),
          Ykt_map, Skt_map, num_elbo_draws, iter_msg, logger);
      num_evals += pathfinder_res.first.fn_calls;
      print_log_remainder(write_log_cond, msg, ret, num_evals, lbfgs,
                          pathfinder_res.first.elbo, pathfinder_res.first.elbo,
                          lbfgs_ss, logger);
      if (unlikely(save_iterations)) {
        diagnostic_writer.write("lbfgs_success", true);
        diagnostic_writer.write("pathfinder_success", true);
        diagnostic_writer.write("x_center", pathfinder_res.second.x_center);
        diagnostic_writer.write("logDetCholHk",
                                pathfinder_res.second.logdetcholHk);
        diagnostic_writer.write("L_approx", pathfinder_res.second.L_approx);
        diagnostic_writer.write("Qk", pathfinder_res.second.Qk);
        diagnostic_writer.write("alpha", pathfinder_res.second.alpha);
        diagnostic_writer.write("full", pathfinder_res.second.use_full);
        diagnostic_writer.write("lbfgs_note", lbfgs_ss.str());
        diagnostic_writer.end_record();
      }
      if (lbfgs_ss.str().length() > 0) {
        logger.info(lbfgs_ss);
        lbfgs_ss.str("");
      }

      if (pathfinder_res.first.elbo > elbo_best.elbo) {
        elbo_best = std::move(pathfinder_res.first);
        taylor_approx_best = std::move(pathfinder_res.second);
        best_iteration = lbfgs.iter_num();
      }
    } catch (const std::exception& e) {
      if (unlikely(save_iterations)) {
        diagnostic_writer.write("lbfgs_success", true);
        diagnostic_writer.write("pathfinder_success", false);
        diagnostic_writer.write("history_size", history_size);
        diagnostic_writer.write("history_size", history_size);
        diagnostic_writer.write("lbfgs_note", lbfgs_ss.str());
        diagnostic_writer.write("pathfinder_error", std::string(e.what()));
        diagnostic_writer.end_record();
      }
      if (lbfgs_ss.str().length() > 0) {
        logger.info(lbfgs_ss);
        lbfgs_ss.str("");
      }
      if (ReturnLpSamples) {
        // we want to terminate multi-path pathfinder during these unrecoverable
        // exceptions
        throw;
      } else {
        logger.error(e.what());
        return internal::ret_pathfinder<ReturnLpSamples>(
            error_codes::SOFTWARE, Eigen::Array<double, Eigen::Dynamic, 1>(0),
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(0, 0), 0);
      }
    }
  }
  if (unlikely(save_iterations)) {
    diagnostic_writer.end_record();
  }
  if (unlikely(ret <= 0)) {
    std::string prefix_err_msg
        = "Optimization terminated with error: " + lbfgs.get_code_string(ret);
    if (lbfgs.iter_num() < 2) {
      logger.error(
          prefix_err_msg
          + " Optimization failed to start, pathfinder cannot be run.");
      return internal::ret_pathfinder<ReturnLpSamples>(
          error_codes::SOFTWARE, Eigen::Array<double, Eigen::Dynamic, 1>(0),
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(0, 0),
          std::atomic<size_t>{num_evals + lbfgs.grad_evals()});
    } else {
      logger.warn(prefix_err_msg +
          " Stan will still attempt pathfinder but may fail or produce "
          "incorrect results.");
    }
  }
  if (unlikely(best_iteration == -1)) {
    logger.error(path_num +
        "Failure: None of the LBFGS iterations completed "
        "successfully");
    return internal::ret_pathfinder<ReturnLpSamples>(
        error_codes::SOFTWARE, Eigen::Array<double, Eigen::Dynamic, 1>(0),
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(0, 0), num_evals);
  } else {
    if (refresh != 0) {
      logger.info(path_num + "Best Iter: [" + std::to_string(best_iteration)
                  + "] ELBO (" + std::to_string(elbo_best.elbo) + ")"
                  + " evaluations: (" + std::to_string(num_evals) + ")");
    }
  }
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> constrained_draws_mat;
  Eigen::Array<double, Eigen::Dynamic, 1> lp_ratio;
  auto&& elbo_draws = elbo_best.repeat_draws;
  auto&& elbo_lp_ratio = elbo_best.lp_ratio;
  auto&& elbo_lp_mat = elbo_best.lp_mat;
  const int remaining_draws = num_draws - elbo_lp_ratio.rows();
  const Eigen::Index num_unconstrained_params = names.size() - 2;
  if (likely(remaining_draws > 0)) {
    try {
      internal::elbo_est_t est_draws = internal::est_approx_draws<false>(
          lp_fun, constrain_fun, rng, taylor_approx_best, remaining_draws,
          taylor_approx_best.alpha, path_num, logger, calculate_lp);
      num_evals += est_draws.fn_calls;
      auto&& new_lp_ratio = est_draws.lp_ratio;
      auto&& lp_draws = est_draws.lp_mat;
      auto&& new_draws = est_draws.repeat_draws;
      lp_ratio = Eigen::Array<double, Eigen::Dynamic, 1>(elbo_lp_ratio.size()
                                                         + new_lp_ratio.size());
      lp_ratio.head(elbo_lp_ratio.size()) = elbo_lp_ratio.array();
      lp_ratio.tail(new_lp_ratio.size()) = new_lp_ratio.array();
      const auto total_size = elbo_draws.cols() + new_draws.cols();
      constrained_draws_mat
          = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(names.size(),
                                                                  total_size);
      Eigen::VectorXd unconstrained_col;
      Eigen::VectorXd approx_samples_constrained_col;
      for (Eigen::Index i = 0; i < elbo_draws.cols(); ++i) {
        constrained_draws_mat.col(i).head(2) = elbo_lp_mat.row(i).matrix();
        unconstrained_col = elbo_draws.col(i);
        constrained_draws_mat.col(i).tail(num_unconstrained_params)
            = constrain_fun(rng, unconstrained_col,
                            approx_samples_constrained_col)
                  .matrix();
      }
      for (Eigen::Index i = elbo_draws.cols(), j = 0; i < total_size;
           ++i, ++j) {
        constrained_draws_mat.col(i).head(2) = lp_draws.row(j).matrix();
        unconstrained_col = new_draws.col(j);
        constrained_draws_mat.col(i).tail(num_unconstrained_params)
            = constrain_fun(rng, unconstrained_col,
                            approx_samples_constrained_col)
                  .matrix();
      }
    } catch (const std::domain_error& e) {
      std::string err_msg = e.what();
      logger.warn(path_num + "Final sampling approximation failed with error: "
                  + err_msg);
      logger.info(
          path_num
          + "Returning the approximate samples used for ELBO calculation: "
          + err_msg);
      constrained_draws_mat
          = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(
              names.size(), elbo_draws.cols());
      Eigen::VectorXd approx_samples_constrained_col;
      Eigen::VectorXd unconstrained_col;
      for (Eigen::Index i = 0; i < elbo_draws.cols(); ++i) {
        constrained_draws_mat.col(i).head(2) = elbo_lp_mat.row(i).matrix();
        unconstrained_col = elbo_draws.col(i);
        constrained_draws_mat.col(i).tail(num_unconstrained_params)
            = constrain_fun(rng, unconstrained_col,
                            approx_samples_constrained_col)
                  .matrix();
      }
      lp_ratio = std::move(elbo_best.lp_ratio);
    }
  } else {
    // output only first num_draws from what we computed for ELBO
    constrained_draws_mat
        = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(names.size(),
                                                                num_draws);
    Eigen::VectorXd approx_samples_constrained_col;
    Eigen::VectorXd unconstrained_col;
    for (Eigen::Index i = 0; i < num_draws; ++i) {
      constrained_draws_mat.col(i).head(2) = elbo_lp_mat.row(i).matrix();
      unconstrained_col = elbo_draws.col(i);
      constrained_draws_mat.col(i).tail(num_unconstrained_params)
          = constrain_fun(rng, unconstrained_col,
                          approx_samples_constrained_col)
                .matrix();
    }
    lp_ratio = std::move(elbo_best.lp_ratio.head(num_draws));
  }
  parameter_writer(constrained_draws_mat);
  parameter_writer();
  const auto end_pathfinder_time = std::chrono::steady_clock::now();
  const double pathfinder_delta_time = stan::services::util::duration_diff(
      start_pathfinder_time, end_pathfinder_time);
  std::string pathfinder_time_str = "Elapsed Time: ";
  pathfinder_time_str += std::to_string(pathfinder_delta_time)
                         + std::string(" seconds (Pathfinder)");
  parameter_writer(pathfinder_time_str);
  parameter_writer();
  return internal::ret_pathfinder<ReturnLpSamples>(
      error_codes::OK, std::move(lp_ratio), std::move(constrained_draws_mat),
      num_evals);
}

}  // namespace pathfinder
}  // namespace services
}  // namespace stan
#endif
