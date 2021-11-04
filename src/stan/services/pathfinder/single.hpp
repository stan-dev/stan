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
inline auto crossprod(T1&& x, T2&& y) {
  return x.transpose() * y;
}

template <typename T1>
inline auto crossprod(T1&& x) {
  return x.transpose() * x;
}

template <typename T1, typename T2>
inline auto tcrossprod(T1&& x, T2&& y) {
  return x * y.transpose();
}

template <typename T1>
inline auto tcrossprod(T1&& x) {
  return x * x.transpose();
}

inline bool is_nan(double x) {
  return x == std::numeric_limits<double>::quiet_NaN();
}

inline bool is_infinite(double x) {
  return x == std::numeric_limits<double>::infinity();
}

inline auto form_init_diag(const Eigen::Array<double, -1, 1>& E0,
                           const Eigen::Array<double, -1, 1>& Yk,
                           const Eigen::Array<double, -1, 1>& Sk) {
  double Dk = (Yk * Sk).sum();
  auto yk_sq = Yk.square().eval();
  double thetak = yk_sq.sum() / Dk;
  double a = ((E0 * yk_sq).sum() / Dk);
  return 1.0
         / (a / E0 + yk_sq / Dk
            - a * (Sk / E0).square() / (Sk.square() / E0).sum());
}

struct taylor_approx_t {
  Eigen::VectorXd x_center;
  double logdetcholHk;
  Eigen::MatrixXd L_approx;
  Eigen::MatrixXd Qk;
  bool use_full;
};

struct div_est_t {
  double DIV{-420};
  int fn_calls_DIV;
  Eigen::MatrixXd repeat_draws;
  Eigen::VectorXd fn_draws;
  Eigen::VectorXd lp_approx_draws;
};

template <typename Generator>
inline auto calc_u_u2(Generator& rnorm, const taylor_approx_t& taylor_approx,
                      const Eigen::VectorXd& alpha) {
  if (taylor_approx.use_full) {
    Eigen::MatrixXd u = rnorm().eval();
    Eigen::MatrixXd u2 = (crossprod(taylor_approx.L_approx, u).colwise()
                          + taylor_approx.x_center)
                             .eval();
    return std::make_tuple(std::move(u), std::move(u2));
  } else {
    Eigen::MatrixXd u = rnorm().eval();
    Eigen::MatrixXd u1 = crossprod(taylor_approx.Qk, u).eval();
    Eigen::MatrixXd u22
        = (taylor_approx.Qk * crossprod(taylor_approx.L_approx, u1)
           + (u - taylor_approx.Qk * u1));
    Eigen::MatrixXd u2
        = ((alpha.array().inverse().sqrt().matrix().asDiagonal() * u22)
               .colwise()
           + taylor_approx.x_center)
              .eval();
    return std::make_tuple(std::move(u), std::move(u2));
  }
}

template <typename SamplePkg, typename F, typename BaseRNG>
auto est_DIV(const SamplePkg& taylor_approx, size_t num_samples,
             const Eigen::VectorXd& alpha, F&& fn, const char* label,
             BaseRNG& rng) {
  const auto D = taylor_approx.x_center.size();
  int draw_ind = 1;
  int fn_calls_DIV = 0;
  boost::variate_generator<BaseRNG&, boost::normal_distribution<>>
      rand_unit_gaus(rng, boost::normal_distribution<>());
  auto rnorm = [&rand_unit_gaus, D, num_samples]() {
    return Eigen::MatrixXd::NullaryExpr(
        D, num_samples, [&rand_unit_gaus]() { return rand_unit_gaus(); });
  };
  auto tuple_u = calc_u_u2(rnorm, taylor_approx, alpha);
  auto&& u = std::get<0>(tuple_u);
  auto&& u2 = std::get<1>(tuple_u);
  // skip bad samples
  Eigen::VectorXd f_test_DIV(u2.cols());
  try {
    for (Eigen::Index i = 0; i < f_test_DIV.cols(); ++i) {
      f_test_DIV(i) = fn(u2.col(i).eval());
    }
  } catch (...) {
    // TODO: Actually catch errors
  }
  Eigen::VectorXd fn_draws = f_test_DIV;
  Eigen::VectorXd lp_approx_draws
      = -taylor_approx.logdetcholHk - 0.5 * u.array().square().colwise().sum()
        - 0.5 * D * log(2 * 3.14);  // NOTE THIS NEEDS TO BE pi()
  Eigen::MatrixXd repeat_draws = u2;
  //### Divergence estimation ###
  double ELBO = -fn_draws.mean() - lp_approx_draws.mean();
  if (is_nan(ELBO)) {
    ELBO = -std::numeric_limits<double>::infinity();
  }
  return div_est_t{ELBO, fn_calls_DIV, std::move(repeat_draws),
                   std::move(fn_draws), std::move(lp_approx_draws)};
}

template <typename SamplePkg, typename BaseRNG>
inline auto approximation_samples(const SamplePkg& taylor_approx,
                                  size_t num_samples,
                                  const Eigen::VectorXd& alpha, BaseRNG&& rng) {
  const Eigen::Index num_params = taylor_approx.x_center.size();
  boost::variate_generator<BaseRNG&, boost::normal_distribution<>>
      rand_unit_gaus(rng, boost::normal_distribution<>());
  auto rnorm = [&rand_unit_gaus, num_params, num_samples]() {
    return Eigen::MatrixXd::NullaryExpr(
        num_params, num_samples,
        [&rand_unit_gaus]() { return rand_unit_gaus(); });
  };
  auto tuple_u = calc_u_u2(rnorm, taylor_approx, alpha);
  using map_t = Eigen::Map<Eigen::MatrixXd>;
  Eigen::VectorXd lp_approx_draws
      = -taylor_approx.logdetcholHk
        - 0.5
              * map_t(std::get<0>(tuple_u).data(), num_params, num_samples)
                    .array()
                    .square()
                    .colwise()
                    .sum()
        - 0.5 * num_params * log(2 * stan::math::pi());  // TODO: PUT BACK pi()
  Eigen::MatrixXd final_params(num_params + 1, num_samples);
  final_params.block(0, 0, num_params, num_samples) = std::get<1>(tuple_u);
  final_params.row(num_params) = lp_approx_draws;
  return final_params;
}

inline auto construct_taylor_approximation_full(
    const Eigen::MatrixXd& Ykt_matt, const Eigen::VectorXd& alpha,
    const Eigen::VectorXd& Dk, const Eigen::MatrixXd& ninvRST,
    const Eigen::VectorXd& point_est, const Eigen::VectorXd& grad_est) {
  auto& Ykt_mat = Ykt_matt.transpose();
  /*
  Eigen::MatrixXd ykt_mat_cprod
      = Eigen::MatrixXd(Ykt_mat.rows(), Ykt_mat.cols())
            .setZero()
            .template selfadjointView<Eigen::Upper>()
            .rankUpdate(Ykt_mat *
  alpha.head(Ykt_mat.cols()).array().sqrt().matrix().asDiagonal());
  ykt_mat_cprod += Dk.asDiagonal();
  */
  Eigen::MatrixXd blah = tcrossprod(
      Ykt_mat
      * alpha.head(Ykt_mat.cols()).array().sqrt().matrix().asDiagonal());
  blah += Dk.asDiagonal();
  Eigen::MatrixXd Hk
      = crossprod(Ykt_mat * alpha.head(Ykt_mat.cols()).asDiagonal(), ninvRST)
        + crossprod(ninvRST, Ykt_mat * alpha.head(Ykt_mat.cols()).asDiagonal())
        + crossprod(ninvRST, (blah)*ninvRST);
  Hk += alpha.asDiagonal();
  Eigen::MatrixXd cholHk = Hk.llt().matrixL();
  auto logdetcholHk = 2.0 * std::log(cholHk.determinant());

  Eigen::VectorXd x_center = point_est - Hk * grad_est;
  return taylor_approx_t{std::move(x_center), logdetcholHk, std::move(cholHk),
                         Eigen::MatrixXd(0, 0), true};
}

inline auto construct_taylor_approximation_sparse(
    const Eigen::MatrixXd& Ykt_matt, const Eigen::VectorXd& alpha,
    const Eigen::VectorXd& Dk, const Eigen::MatrixXd& ninvRST,
    const Eigen::VectorXd& point_est, const Eigen::VectorXd& grad_est) {
  auto Ykt_mat = Ykt_matt.transpose().eval();
  const Eigen::Index m = Ykt_mat.rows();
  Eigen::MatrixXd Wkbart(Ykt_mat.rows() + ninvRST.rows(), Ykt_mat.cols());
  Wkbart.topRows(Ykt_mat.rows())
      = Ykt_mat * alpha.array().sqrt().matrix().asDiagonal();
  Wkbart.bottomRows(ninvRST.rows())
      = ninvRST * alpha.array().inverse().sqrt().matrix().asDiagonal();

  Eigen::MatrixXd Mkbar(2 * m, 2 * m);
  Mkbar.topLeftCorner(m, m).setZero();
  Mkbar.topRightCorner(m, m) = Eigen::DiagonalMatrix<double, -1, -1>(m);
  Mkbar.bottomLeftCorner(m, m) = Eigen::DiagonalMatrix<double, -1, -1>(m);
  Eigen::MatrixXd blah = tcrossprod(
      Ykt_mat
      * alpha.head(Ykt_mat.cols()).array().sqrt().matrix().asDiagonal());
  blah += Dk.asDiagonal();
  Mkbar.bottomRightCorner(m, m) = blah;

  Eigen::HouseholderQR<Eigen::MatrixXd> qr(Wkbart.rows(), Wkbart.cols());
  qr.compute(Wkbart);
  const auto min_size = std::min(Wkbart.rows(), Wkbart.cols());
  Eigen::MatrixXd Rkbar = qr.matrixQR().topLeftCorner(min_size, Wkbart.cols());
  for (int i = 0; i < min_size; i++) {
    for (int j = 0; j < i; j++) {
      Rkbar.coeffRef(i, j) = 0.0;
    }
    if (Rkbar(i, i) < 0) {
      Rkbar.row(i) *= -1.0;
    }
  }
  Eigen::MatrixXd Qk
      = qr.householderQ() * Eigen::MatrixXd::Identity(Wkbart.rows(), min_size);
  for (int i = 0; i < min_size; i++) {
    if (qr.matrixQR().coeff(i, i) < 0) {
      Qk.col(i) *= -1.0;
    }
  }
  Eigen::MatrixXd L_approx
      = (Rkbar * Mkbar * Rkbar.transpose()
         + Eigen::MatrixXd::Identity(Rkbar.rows(), Rkbar.rows()))
            .llt()
            .matrixL();
  double logdetcholHk = L_approx.diagonal().array().log().sum()
                        + 0.5 * alpha.array().log().sum();
  Eigen::VectorXd ninvRSTg = ninvRST * grad_est;
  Eigen::VectorXd x_center_tmp
      = (alpha.array() * grad_est.array()).matrix()
        + (alpha.array() * (Ykt_mat.transpose() * ninvRSTg).array()).matrix()
        + ninvRSTg.transpose() * Ykt_mat
              * (alpha.array() * grad_est.array()).matrix()
        + ninvRSTg.transpose() * blah * ninvRSTg.transpose();
  Eigen::VectorXd x_center = point_est - x_center_tmp;

  return taylor_approx_t{std::move(x_center), logdetcholHk, std::move(L_approx),
                         std::move(Qk), false};
}

inline auto construct_taylor_approximation(const Eigen::MatrixXd& Ykt_mat,
                                           const Eigen::VectorXd& alpha,
                                           const Eigen::VectorXd& Dk,
                                           const Eigen::MatrixXd& ninvRST,
                                           const Eigen::VectorXd& point_est,
                                           const Eigen::VectorXd& grad_est) {
  // If twice the current history size is larger than the number of params
  // use a sparse approximation
  if (2 * Ykt_mat.size() > Ykt_mat.rows()) {
    return construct_taylor_approximation_full(Ykt_mat, alpha, Dk, ninvRST,
                                               point_est, grad_est);
  } else {
    return construct_taylor_approximation_sparse(Ykt_mat, alpha, Dk, ninvRST,
                                                 point_est, grad_est);
  }
}

inline bool check_cond(const Eigen::VectorXd& Yk, const Eigen::VectorXd& Sk) {
  double Dk = (Yk.array() * Sk.array()).sum();
  if (Dk == 0) {
    return false;
  } else {
    double thetak = Yk.array().square().sum() / Dk;
    // curvature checking
    if ((Dk <= 0 || std::abs(thetak) > 1e12)) {  // 2.2*e^{-16}
      return false;
    } else {
      return true;
    }
  }
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
  // callbacks::writer& diagnostic_writer) {
  boost::ecuyer1988 rng = util::create_rng(random_seed, path);

  std::vector<int> disc_vector;
  // 1. Sample initial parameters
  std::vector<double> cont_vector = util::initialize<false>(
      model, init, rng, init_radius, false, logger, init_writer);
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
  lbfgs_iters.reserve(num_iterations + 1);
  int ret = 0;
  {
    std::vector<double> g1;
    double blah = stan::model::log_prob_grad<false, false>(model, cont_vector,
                                                           disc_vector, g1);
    lbfgs_iters.emplace_back(
        Eigen::Map<Eigen::VectorXd>(cont_vector.data(), cont_vector.size()),
        Eigen::Map<Eigen::VectorXd>(g1.data(), g1.size()));
  }
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
    std::vector<double> values;
    std::stringstream msg;
    model.write_array(rng, cont_vector, disc_vector, values, true, true, &msg);
    if (msg.str().length() > 0) {
      logger.info(msg);
    }
    Eigen::VectorXd value_vec
        = Eigen::Map<Eigen::VectorXd>(values.data(), values.size());
    lbfgs_iters.emplace_back(value_vec, lbfgs.curr_g());

    if (save_iterations) {
      values.insert(values.begin(), lp);
      parameter_writer(values);
    }
  }

  // 3. For each L-BFGS iteration `num_iterations`
  Eigen::VectorXd lambda(lbfgs_iters.size() - 1);
  Eigen::VectorXd E = Eigen::VectorXd::Ones(cont_vector.size());
  boost::circular_buffer<Eigen::VectorXd> Ykt_h(history_size);
  boost::circular_buffer<Eigen::VectorXd> Skt_h(history_size);
  double div_max = std::numeric_limits<double>::min();
  div_est_t DIV_fit_best;
  taylor_approx_t taylor_approx_best;
  for (size_t iter = 0; iter < lbfgs_iters.size() - 1; iter++) {
    Eigen::VectorXd Ykt
        = std::get<1>(lbfgs_iters[iter]) - std::get<1>(lbfgs_iters[iter + 1]);
    Eigen::VectorXd Skt
        = std::get<0>(lbfgs_iters[iter]) - std::get<0>(lbfgs_iters[iter + 1]);
    if (iter == 0) {
      Ykt_h.push_back(Ykt);
      Skt_h.push_back(Skt);
    }
    if (check_cond(Ykt, Skt)) {
      // initial estimate of diagonal inverse Hessian
      E = form_init_diag(E.array(), Ykt.array(), Skt.array());
      // update Y and S matrix
      if (iter != 0) {
        Ykt_h.push_back(Ykt);
        Skt_h.push_back(Skt);
      }
    }
    const auto param_size = Ykt.size();
    const auto m = Ykt_h.size();
    Eigen::VectorXd Dk(m);
    for (Eigen::Index i = 0; i < m; i++) {
      Dk[i] = Ykt_h[i].dot(Skt_h[i]);
    }
    Eigen::MatrixXd Rk = Eigen::MatrixXd::Zero(m, m);
    for (Eigen::Index s = 0; s < m; s++) {
      for (Eigen::Index i = 0; i <= s; i++) {
        Rk(i, s) = Skt_h[i].dot(Ykt_h[s]);
      }
    }
    Eigen::MatrixXd ninvRST;
    {
      Eigen::MatrixXd Skt_mat(param_size, m);
      for (Eigen::Index i = 0; i < m; ++i) {
        Skt_mat.col(i) = Skt_h[i];
      }
      ninvRST = -(Rk.triangularView<Eigen::Upper>().solve(Skt_mat.transpose()));
    }

    Eigen::MatrixXd Ykt_mat(param_size, m);
    for (Eigen::Index i = 0; i < m; ++i) {
      Ykt_mat.col(i) = Ykt_h[i];
    }
    bool ill_dist = false;
    /**
     * 3a. Run BFGS-Sample to get `num_elbo_draws` draws from normal
     * approximation and log density of draws in the approximate normal
     * distribution
     */
    taylor_approx_t taylor_appx_tuple = construct_taylor_approximation(
        Ykt_mat, E, Dk, ninvRST, std::get<0>(lbfgs_iters[iter]),
        std::get<1>(lbfgs_iters[iter]));
    bool ill_distr = false;
    if (ill_distr) {  // Ignore this for now ||
                      // stan::math::is_nan(std::get<0>(taylor_approx[1]))) {
      continue;
    }
    auto fn = [&model, &disc_vector](auto&& u) {
      return model.template log_prob<false, false>(
          const_cast<std::decay_t<decltype(u)>&>(u), 0);
    };
    //#lCDIV #lADIV  #lIKL #ELBO
    auto DIV_fit
        = est_DIV(taylor_appx_tuple, num_elbo_draws, E, fn, "ELBO", rng);
    // TODO: Calculate total function calls
    // fn_call = fn_call + DIV_fit$fn_calls_DIV
    // DIV_ls = c(DIV_ls, DIV_fit$DIV)
    //  4. Find $l \in L$ that maximizes ELBO $l^* = arg max_l ELBO^(l)$.
    if (DIV_fit.DIV > div_max) {
      div_max = DIV_fit.DIV;
      DIV_fit_best = DIV_fit;
      taylor_approx_best = taylor_appx_tuple;
    }
  }

  // Generate upto num_samples samples from the best approximating Normal ##
  auto draws_N_apx
      = approximation_samples(taylor_approx_best, num_draws, E, rng);
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
