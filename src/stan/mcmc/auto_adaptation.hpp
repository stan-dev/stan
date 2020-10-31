#ifndef STAN_MCMC_AUTO_ADAPTATION_HPP
#define STAN_MCMC_AUTO_ADAPTATION_HPP

#include <stan/math.hpp>
#include <stan/mcmc/windowed_adaptation.hpp>
#include <vector>

namespace stan {

namespace mcmc {
template <typename Model>
struct log_prob_wrapper_covar {
  const Model& model_;
  explicit log_prob_wrapper_covar(const Model& model) : model_(model) {}

  template <typename T>
  T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& q) const {
    return model_.template log_prob<true, true, T>(
        const_cast<Eigen::Matrix<T, Eigen::Dynamic, 1>&>(q), &std::cout);
  }
};

namespace internal {
/**
 * Compute the covariance of data in Y.
 *
 * Columns of Y are different variables. Rows are different samples.
 *
 * When there is only one row in Y, return a covariance matrix of the expected
 *  size filled with zeros.
 *
 * @param Y Data
 * @return Covariance of Y
 */
Eigen::MatrixXd covariance(const Eigen::MatrixXd& Y) {
  stan::math::check_nonzero_size("covariance", "Y", Y);

  Eigen::MatrixXd centered = Y.rowwise() - Y.colwise().mean();
  return centered.transpose() * centered / std::max(centered.rows() - 1.0, 1.0);
}

/**
 * Compute the largest magnitude eigenvalue of a symmetric matrix using the
 * power method. The function f should return the product of that matrix with an
 * abitrary vector.
 *
 * f should take one Eigen::VectorXd argument, x, and return the product of a
 * matrix with x as an Eigen::VectorXd argument of the same size.
 *
 * The eigenvalue is estimated iteratively. If the kth estimate is e_k, then the
 * function returns when either abs(e_{k + 1} - e_k) < tol * abs(e_k) or the
 * maximum number of iterations have been performed
 *
 * This means the returned eigenvalue might not be computed to full precision
 *
 * @param initial_guess Initial guess of the eigenvector of the largest
 * eigenvalue
 * @param[in,out] max_iterations Maximum number of power iterations, on return
 * number of iterations used
 * @param[in,out] tol Relative tolerance, on return the relative error in the
 * eigenvalue estimate
 * @return Largest magnitude eigenvalue of operator f
 */
template <typename F>
double power_method(F& f, const Eigen::VectorXd& initial_guess,
                    int& max_iterations, double& tol) {
  Eigen::VectorXd v = initial_guess;
  double eval = 0.0;
  Eigen::VectorXd Av = f(v);
  stan::math::check_matching_sizes("power_method", "matrix vector product", Av,
                                   "vector", v);

  int i = 0;
  for (; i < max_iterations; ++i) {
    double v_norm = v.norm();
    double new_eval = v.dot(Av) / (v_norm * v_norm);
    if (i == max_iterations - 1
        || std::abs(new_eval - eval) <= tol * std::abs(eval)) {
      tol = std::abs(new_eval - eval) / std::abs(eval);
      eval = new_eval;
      max_iterations = i + 1;
      break;
    }

    eval = new_eval;
    v = Av / Av.norm();

    Av = f(v);
  }

  return eval;
}

/**
 * Compute the largest eigenvalue of the sample covariance rescaled by a metric,
 *  that is, the largest eigenvalue of L^{-1} \Sigma L^{-T}
 *
 * @param L Cholesky decomposition of Metric
 * @param Sigma Sample covariance
 * @return Largest eigenvalue
 */
double eigenvalue_scaled_covariance(const Eigen::MatrixXd& L,
                                    const Eigen::MatrixXd& Sigma) {
  Eigen::MatrixXd S = L.template triangularView<Eigen::Lower>()
                          .solve(L.template triangularView<Eigen::Lower>()
                                     .solve(Sigma)
                                     .transpose())
                          .transpose();

  auto Sx = [&](Eigen::VectorXd x) -> Eigen::VectorXd { return S * x; };

  int max_iterations = 200;
  double tol = 1e-5;

  return internal::power_method(Sx, Eigen::VectorXd::Random(Sigma.cols()),
                                max_iterations, tol);
}

/**
 * Compute the largest eigenvalue of the Hessian of the log density rescaled by
 * a metric, that is, the largest eigenvalue of L^T \nabla^2_{qq} H(q) L
 *
 * @tparam Model Type of model
 * @param model Defines the log density
 * @param q Point around which to compute the Hessian
 * @param L Cholesky decomposition of Metric
 * @return Largest eigenvalue
 */
template <typename Model>
double eigenvalue_scaled_hessian(const Model& model, const Eigen::MatrixXd& L,
                                 const Eigen::VectorXd& q) {
  Eigen::VectorXd eigenvalues;
  Eigen::MatrixXd eigenvectors;

  auto hessian_vector = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
    double lp;
    Eigen::VectorXd grad1;
    Eigen::VectorXd grad2;
    // stan::math::hessian_times_vector(log_prob_wrapper_covar<Model>(model), q,
    // x, lp, Ax);
    double dx = 1e-5;
    Eigen::VectorXd dr = L * x * dx;
    stan::math::gradient(log_prob_wrapper_covar<Model>(model), q + dr / 2.0, lp,
                         grad1);
    stan::math::gradient(log_prob_wrapper_covar<Model>(model), q - dr / 2.0, lp,
                         grad2);
    return L.transpose() * (grad1 - grad2) / dx;
  };

  int max_iterations = 200;
  double tol = 1e-5;

  return internal::power_method(
      hessian_vector, Eigen::VectorXd::Random(q.size()), max_iterations, tol);
}
}  // namespace internal

class auto_adaptation : public windowed_adaptation {
 public:
  explicit auto_adaptation(int n) : windowed_adaptation("covariance") {}
  /**
   * Update the metric if at the end of an adaptation window.
   *
   * @tparam Model Type of model
   * @param model Defines the log density
   * @param covar[out] New metric
   * @param covar_is_diagonal[out] Set to true if metric is diagonal, false
   * otherwise
   * @param q New MCMC draw
   * @return True if this was the end of an adaptation window, false otherwise
   */
  template <typename Model>
  bool learn_covariance(const Model& model, Eigen::MatrixXd& covar,
                        bool& covar_is_diagonal, const Eigen::VectorXd& q) {
    if (adaptation_window())
      qs_.push_back(q);

    if (end_adaptation_window()) {
      compute_next_window();

      int M = q.size();
      int N = qs_.size();

      Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(M, N);
      std::vector<int> idxs(N);
      for (int i = 0; i < qs_.size(); i++)
        idxs[i] = i;

      std::random_shuffle(idxs.begin(), idxs.end());
      for (int i = 0; i < qs_.size(); i++)
        Y.block(0, i, M, 1) = qs_[idxs[i]];

      try {
        bool use_dense = false;
        // 's' stands for selection
        // 'r' stands for refinement
        for (char state : {'s', 'r'}) {
          Eigen::MatrixXd Ytrain;
          Eigen::MatrixXd Ytest;

          // If in selection state
          if (state == 's') {
            int Mtest;
            Mtest = static_cast<int>(0.2 * Y.cols());
            if (Mtest < 5) {
              Mtest = 5;
            }

            if (Y.cols() < 10) {
              throw std::runtime_error(
                  "Each warmup stage must have at least 10 samples");
            }

            Ytrain = Y.block(0, 0, M, Y.cols() - Mtest);
            Ytest = Y.block(0, Ytrain.cols(), M, Mtest);
          } else {
            Ytrain = Y;
          }

          Eigen::MatrixXd cov_train
              = (Ytrain.cols() > 0) ? internal::covariance(Ytrain.transpose())
                                    : Eigen::MatrixXd::Zero(M, M);
          Eigen::MatrixXd cov_test
              = (Ytest.cols() > 0) ? internal::covariance(Ytest.transpose())
                                   : Eigen::MatrixXd::Zero(M, M);

          Eigen::MatrixXd dense
              = (N / (N + 5.0)) * cov_train
                + 1e-3 * (5.0 / (N + 5.0))
                      * Eigen::MatrixXd::Identity(cov_train.rows(),
                                                  cov_train.cols());

          Eigen::MatrixXd diag = dense.diagonal().asDiagonal();

          covar = dense;

          // If in selection state
          if (state == 's') {
            Eigen::MatrixXd L_dense = dense.llt().matrixL();
            Eigen::MatrixXd L_diag
                = diag.diagonal().array().sqrt().matrix().asDiagonal();

            double low_eigenvalue_dense
                = -1.0
                  / internal::eigenvalue_scaled_covariance(L_dense, cov_test);
            double low_eigenvalue_diag
                = -1.0
                  / internal::eigenvalue_scaled_covariance(L_diag, cov_test);

            double c_dense = 0.0;
            double c_diag = 0.0;
            for (int i = 0; i < 5; i++) {
              double high_eigenvalue_dense
                  = internal::eigenvalue_scaled_hessian(
                      model, L_dense, Ytest.block(0, i, M, 1));
              double high_eigenvalue_diag = internal::eigenvalue_scaled_hessian(
                  model, L_diag, Ytest.block(0, i, M, 1));

              c_dense = std::max(c_dense, std::sqrt(high_eigenvalue_dense
                                                    / low_eigenvalue_dense));
              c_diag = std::max(c_diag, std::sqrt(high_eigenvalue_diag
                                                  / low_eigenvalue_diag));
            }

            std::cout << "adapt: " << adapt_window_counter_
                      << ", which: dense, max: " << c_dense << std::endl;
            std::cout << "adapt: " << adapt_window_counter_
                      << ", which: diag, max: " << c_diag << std::endl;

            if (c_dense < c_diag) {
              use_dense = true;
            } else {
              use_dense = false;
            }
          } else {
            if (use_dense) {
              covar = dense;
              covar_is_diagonal = false;
            } else {
              covar = diag;
              covar_is_diagonal = true;
            }
          }
        }
      } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        std::cout
            << "Exception while using auto adaptation, falling back to diagonal"
            << std::endl;
        Eigen::MatrixXd cov = internal::covariance(Y.transpose());
        covar = ((N / (N + 5.0)) * cov.diagonal()
                 + 1e-3 * (5.0 / (N + 5.0)) * Eigen::VectorXd::Ones(cov.cols()))
                    .asDiagonal();
        covar_is_diagonal = true;
      }

      ++adapt_window_counter_;
      qs_.clear();

      return true;
    }

    ++adapt_window_counter_;
    return false;
  }

 protected:
  std::vector<Eigen::VectorXd> qs_;
};

}  // namespace mcmc

}  // namespace stan

#endif
