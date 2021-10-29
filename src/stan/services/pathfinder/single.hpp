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

template <typename T1, typename T2>
inline auto crossprod(T1&& x, T2&& y) {
  return x.transpose() * y;
}

template <typename T1, typename T2>
inline auto tcrossprod(T1&& x, T2&& y) {
  return x * y.transpose();
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

struct sample_pkg_t {
  std::string label;
  Eigen::VectorXd x_center;
  Eigen::MatrixXd cholHk;
  Eigen::MatrixXd Qk;
  Eigen::VectorXd theta_D;
  Eigen::MatrixXd Rktilde;
  double logdetcholHk;
};

template <typename Generator>
inline auto calc_u_u2(Generator& rnorm, const sample_full_pkg& sample_pkg,
                      const Eigen::VectorXd& alpha, size_t m) {
  u = rnorm(m);
  u2 = crossprod(sample_pkg.L_approx, u) + sample_pkg.x_center;
  return std::forward_as_tuple(std::move(u), std::move(u2))
}

template <typename Generator>
inline auto calc_u_u2(Generator& rnorm, const sample_sparse_pkg& sample_pkg,
                      const Eigen::VectorXd& alpha, size_t m) {
  u = rnorm(m);
  Eigen::VectorXd u1 = crossprod(sample_pkg.Qk, u);
  u2 = alpha.array().inverse().sqrt().matrix().asDiagonal()
           * (sample_pkg.Qk * crossprod(sample_pkg.Rktilde, u1)
              + (u - sample_pkg.Qk * u1))
       + sample_pkg.x_center;
  return std::forward_as_tuple(std::move(u), std::move(u2))
}

template <typename SamplePkg, typename F, typename BaseRNG>
auto est_DIV(const SamplePkg& sample_pkg, size_t N_sam,
             const Eigen::VectorXd& alpha, F&& fn, std::string label,
             BaseRNG& rng) {
  const auto D = sample_pkg.x_center.size();
  // Should fill with Inf!
  Eigen::VectorXd fn_draws = Eigen::VectorXd::Zero(N_sam);
  Eigen::VectorXd lp_approx_draws = Eigen::VectorXd::Zero(N_sam);
  Eigen::MatrixXd repeat_draws = Eigen::MatrixXd::Zero(D, N_sam);
  int draw_ind = 1;
  int fn_calls_DIV = 0;
  int f_test_DIV = -std::numeric_limits<int>::infinity();
  boost::variate_generator<BaseRNG&, boost::normal_distribution<>>
      rand_unit_gaus(rng, boost::normal_distribution<>());
  std::tuple<Eigen::VectorXd, Eigen::VectorXd> tuple_u;
  auto rnorm = [&rand_unit_gaus](auto D) {
    return Eigen::VectorXd::NullaryExpr(
        D, [&rand_unit_gaus]() { return rand_unit_gaus(); });
  };
  for (int l = 0; l < (2 * N_sam); ++l) {
    auto tuple_u = calc_u_u2(rnorm, sample_pkg, alpha, D);
    auto&& u = std::get<0>(tuple_u);
    auto&& u2 = std::get<1>(tuple_u);
    // skip bad samples
    bool skip_flag = false;
    double f_test_DIV;
    try {
      f_test_DIV = fn(u2);
    } catch (...) {
      // TODO: Actually catch errors
      skip_flag = true;
    }
    if (f_test_DIV == std::numeric_limits<double>::quiet_NaN()) {
      skip_flag = true;
    }
    if (skip_flag) {
      continue;
    } else {
      fn_draws[draw_ind] = f_test_DIV;
      lp_approx_draws[draw_ind]
          = -sample_pkg.logdetcholHk - 0.5 * u.array().square().sum()
            - 0.5 * D * log(2 * 3.14);  // NOTE THIS NEEDS TO BE pi()
      repeat_draws(0, draw_ind) = u2;
      draw_ind = draw_ind + 1;
    }
    fn_calls_DIV = fn_calls_DIV + 1;
    if (draw_ind == N_sam + 1) {
      break;
    }
  }

  //### Divergence estimation ###
  double ELBO = -fn_draws.mean() - lp_approx_draws.mean();
  ;
  if (is_nan(ELBO)) {
    ELBO = -std::numeric_limits<double>::infinity();
  }
  double DIV;
  if (label == "ELBO") {
    DIV = ELBO;
  } else if (label == "lIKL") {
    // log Inclusive-KL E[p(x_i)/q(x_k)]
    DIV = ELBO
          + log((-fn_draws.array() - lp_approx_draws.array() - ELBO)
                    .exp()
                    .mean());
  } else if (label == "lADIV") {
    // log alpha-divergence E[(p(x_i)/q(x_k))^alpha], e.g. with 1/2
    DIV = 0.5 * ELBO
          + log(((0.5 * (-fn_draws.array() - lp_approx_draws.array() - ELBO))
                     .exp()
                     .mean()));
  } else if (label == "lCDIV") {
    // log Chi^2-divergence is alpha-divergence with alpha=2
    DIV = 2.0 * ELBO
          + log(((2.0
                  * (-fn_draws.array() - lp_approx_draws.array() - ELBO)
                        .exp()
                        .mean())));
  } else {
    throw std::domain_error("The divergence is misspecified");
  }

  if (is_nan(DIV)) {
    DIV = -std::numeric_limits<double>::infinity();
  }
  if (is_infinite(DIV)) {
    DIV = -std::numeric_limits<double>::infinity();
  }

  if ((fn_draws.array() == std::numeric_limits<double>::infinity()).any()) {
    fn_draws[0] = std::numeric_limits<double>::infinity();
  }
  /*
  return(list(DIV = DIV,
              repeat_draws = repeat_draws, fn_draws = fn_draws,
              lp_approx_draws = lp_approx_draws, fn_calls_DIV = fn_calls_DIV))
    */
}

template <typename SamplePkg, typename BaseRNG>
inline auto Sam_N_apx(const SamplePkg& sample_pkg, size_t N_sam,
                      BaseRNG&& rng) {
  const Eigen::Index D = sample_pkg.x_center.size();
  Eigen::VectorXd lp_approx_draws = Eigen::VectorXd::Zero(N_sam);
  Eigen::MatrixXd repeat_draws = Eigen::VectorXd::Zero(D, N_sam);

  boost::variate_generator<BaseRNG&, boost::normal_distribution<>>
      rand_unit_gaus(rng, boost::normal_distribution<>());
  auto rnorm = [&rand_unit_gaus](auto D) {
    return Eigen::VectorXd::NullaryExpr(
        D, [&rand_unit_gaus]() { return rand_unit_gaus(); });
  };
  auto tuple_u = calc_u_u2(rnorm, sample_pkg, D * N_sam);
  using map_t = Eigen::Map<Eigen::MatrixXd> Eigen::VectorXd lp_apx_draws
      = -sample_pkg.logdetcholHk
        - 0.5
              * map_t(std::get<0>(tuple_u).data(), D, N_sam)
                    .array()
                    .square()
                    .colwise()
                    .sum()
        - 0.5 * D* log(2 * 3.14);  // TODO: PUT BACK pi()
  return std::make_tuple(std::get<1>(tuple_u), lp_apx_draws);
}

inline auto form_n_apx_taylor_full(const Eigen::MatrixXd& Ykt_mat,
                                   const Eigen::VectorXd& alpha,
                                   const Eigen::VectorXd& Dk,
                                   const Eigen::MatrixXd& ninvRST,
                                   const Eigen::VectorXd& point_est,
                                   const Eigen::VectorXd& grad_est) {
  Eigen::MatrixXd ykt_mat_cprod
      = Eigen::MatrixXd(Ykt_mat.rows(), Ykt_mat.cols())
            .setZero()
            .template selfadjointView<Eigen::Upper>()
            .rankUpdate(Ykt_mat * alpha.array().sqrt().matrix().asDiagonal());
  ykt_mat_cprod += Dk.asDiagonal();
  Eigen::MatrixXd Hk = ((Ykt_mat * alpha.asDiagonal()).transpose() * ninvRST)
                       + (ninvRST.transpose() * Ykt_mat * alpha.asDiagonal())
                       + (ninvRST.transpose() * (ykt_mat_cprod)*ninvRST);
  Hk += alpha.asDiagonal();
  Eigen::MatrixXd cholHk = Hk.llt().matrixL();
  auto logdetcholHk = 2.0 * std::log(cholHk.determinant());

  Eigen::VectorXd x_center = point_est - Hk * grad_est;
  return 1;
}

inline auto form_n_apx_taylor_sparse(const Eigen::MatrixXd& Ykt_mat,
                                     const Eigen::VectorXd& alpha,
                                     const Eigen::VectorXd& Dk,
                                     const Eigen::MatrixXd& ninvRST,
                                     const Eigen::VectorXd& point_est,
                                     const Eigen::VectorXd& grad_est) {
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
  Eigen::MatrixXd ykt_mat_cprod
      = Eigen::MatrixXd(Ykt_mat.rows(), Ykt_mat.cols())
            .setZero()
            .template selfadjointView<Eigen::Upper>()
            .rankUpdate(Ykt_mat * alpha.array().sqrt().matrix().asDiagonal());
  ykt_mat_cprod += Dk.asDiagonal();
  Mkbar.bottomRightCorner(m, m) = ykt_mat_cprod;

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
  Eigen::MatrixXd Rktilde
      = (Rkbar * Mkbar * Rkbar.transpose()
         + Eigen::MatrixXd::Identity(Rkbar.rows(), Rkbar.rows()))
            .llt()
            .matrixL();
  double logdetcholHk = Rktilde.diagonal().array().log().sum()
                        + 0.5 * alpha.array().log().sum();
  Eigen::VectorXd ninvRSTg = ninvRST * grad_est;
  Eigen::VectorXd x_center_tmp
      = (alpha.array() * grad_est.array()).matrix()
        + (alpha.array() * (Ykt_mat.transpose() * ninvRSTg).array()).matrix()
        + ninvRSTg.transpose() * Ykt_mat
              * (alpha.array() * grad_est.array()).matrix()
        + ninvRSTg.transpose() * ykt_mat_cprod * ninvRSTg.transpose();
  Eigen::VectorXd x_center = point_est - x_center_tmp;

  return std::make_tuple();
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
  std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> lbfgs_iters;
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
    std::vector<double> values;
    std::stringstream msg;
    model.write_array(rng, cont_vector, disc_vector, values, true, true, &msg);
    if (msg.str().length() > 0) {
      logger.info(msg);
    }

    lbfgs_iters.emplace_back(values, lbfgs.curr_g());

    if (save_iterations) {
      values.insert(values.begin(), lp);
      parameter_writer(values);
    }
  }

  // 3. For each L-BFGS iteration `num_iterations`
  Eigen::VectorXd lambda(lbfgs_iters.size());
  Eigen::VectorXd E = Eigen::VectorXd::Ones();
  boost::circular_buffer<Eigen::VectorXd> Ykt_h(history_size);
  boost::circular_buffer<Eigen::VectorXd> Skt_h(history_size);
  double div_max = std::numeric_limits<double>::min();
  for (size_t iter = 0; iter < lbfgs_iters; iter++) {
    Eigen::VectorXd Ykt
        = std::get<1>(lbfgs_iters[i]) - std::get<1>(lbfgs_iters[i + 1]);
    Eigen::VectorXd Skt
        = std::get<0>(lbfgs_iters[i]) - std::get<0>(lbfgs_iters[i + 1]);
    if (check_cond(Ykt, Skt)) {
      // initial estimate of diagonal inverse Hessian
      E = form_init_diag(E.array(), Ykt.array(), Skt.array());
      // update Y and S matrix
      Ykt_h.push_back(Ykt);
      Skt_h.push_back(Skt);
    }
    const auto param_size = Ykt.size();
    const auto m = Ykt_h.size();
    Eigen::VectorXd Dk(m);
    Eigen::VectorXd thetak(m);
    for (Eigen::Index i = 0; i < m; i++) {
      Dk[i] = Ykt_h[i].dot(Skt_h[i]);
      thetak[i]
          = Ykt_h[i].array().square().sum() / Dk[i];  // curvature checking
    }

    Eigen::MatrixXd Rk = Eigen::MatrixXd::Zero(m, m);
    for (Eigen::Index s = 0; s > m; s++) {
      for (Eigen::Index i = 0; i > s; i++) {
        Rk(i, s) = Skt_h[i].dot(Ykt_h[s]);
      }
    }
    Eigen::MatrixXd ninvRST;

    {
      Eigen::MatrixXd Skt_mat(param_size, m);
      for (Eigen::Index i = 0; i < m; ++i) {
        Skt_mat.col(i) = Skt_h[i];
      }
      ninvRST = -(Rk.template triangularView<Eigen::Upper>().solve(Skt_mat));
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

    if (2 * m >= param_size) {
      full_ret_t tayler_appx_tuple form_n_apx_taylor_full(
          Ykt_mat, alpha, Dk, ninvRST, std::get<0>(lbfgs_iters[i]),
          std::get<1>(lbfgs_iters[i]));
    } else {
      sparse_ret_t tayler_appx_tuple form_n_apx_taylor_sparse(
          Ykt_mat, alpha, Dk, ninvRST, std::get<1>(lbfgs_iters[i]),
          std::get<1>(lbfgs_iters[i]));
    }
    if (ill_distr) {
      continue;
    }
    if (is_na(sample_pkg[1])) {
      continue;
    }
    DIV_fit = est_DIV(sample_pkg, N_sam_DIV, fn,
                      label = "ELBO")  //#lCDIV #lADIV  #lIKL #ELBO
        // TODO: Calculate total function calls
        // fn_call = fn_call + DIV_fit$fn_calls_DIV
        // DIV_ls = c(DIV_ls, DIV_fit$DIV)
        //  4. Find $l \in L$ that maximizes ELBO $l^* = arg max_l ELBO^(l)$.
        if (DIV_fit.DIV_ > DIV_max) {
      DIV_max = DIV_fit$DIV DIV_fit_pick = DIV_fit sample_pkg_pick = sample_pkg
    }
  }
  /**
   * 5. Run bfgs-Sample to return `num_draws` draws from ELBO-maximizing normal
   * approx and log density of draws in ELBO-maximizing normal approximation.
   */
  std::tuple<Eigen::MatrixXd, Eigen::VectorXd> final_samples
      = Sam_N_apx(sample_pkg_pick, num_draws, rng);
  return return_code;
}

}  // namespace optimize
}  // namespace services
}  // namespace stan
#endif
