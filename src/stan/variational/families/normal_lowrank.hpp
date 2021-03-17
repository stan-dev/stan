#ifndef STAN_VARIATIONAL_NORMAL_LOWRANK_HPP
#define STAN_VARIATIONAL_NORMAL_LOWRANK_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/math/prim.hpp>
#include <stan/model/gradient.hpp>
#include <stan/variational/base_family.hpp>
#include <algorithm>
#include <ostream>
#include <vector>

namespace stan {
namespace variational {
class normal_lowrank : public base_family {
 private:
  Eigen::VectorXd mu_;
  Eigen::MatrixXd B_;
  Eigen::VectorXd log_d_;

  const int dimension_;
  const int rank_;

  void validate_mean(const char* function, const Eigen::VectorXd& mu) {
    stan::math::check_not_nan(function, "Mean vector", mu);
    stan::math::check_size_match(function, "Dimension of input vector",
                                 mu.size(), "Dimension of current vector",
                                 dimension());
  }

  void validate_factor(const char* function, const Eigen::MatrixXd& B) {
    stan::math::check_not_nan(function, "Low rank factor", B);
    stan::math::check_size_match(function, "Dimension of mean vector",
                                 dimension(), "Dimension of low-rank factor",
                                 B.rows());
    stan::math::check_size_match(function, "Rank of factor", B.cols(),
                                 "Rank of approximation", rank());
  }

  void validate_noise(const char* function, const Eigen::VectorXd& log_d) {
    stan::math::check_not_nan(function, "log std vector", log_d);
    stan::math::check_size_match(function, "Dimension of mean vector",
                                 dimension(), "Dimension of log std vector",
                                 log_d.size());
  }

 public:
  explicit normal_lowrank(const Eigen::VectorXd& mu, size_t rank)
      : mu_(mu),
        B_(Eigen::MatrixXd::Zero(mu.size(), rank)),
        log_d_(Eigen::VectorXd::Zero(mu.size())),
        dimension_(mu.size()),
        rank_(rank) {}

  explicit normal_lowrank(size_t dimension, size_t rank)
      : mu_(Eigen::VectorXd::Zero(dimension)),
        B_(Eigen::MatrixXd::Zero(dimension, rank)),
        log_d_(Eigen::VectorXd::Zero(dimension)),
        dimension_(dimension),
        rank_(rank) {}

  explicit normal_lowrank(const Eigen::VectorXd& mu, const Eigen::MatrixXd& B,
                          const Eigen::VectorXd& log_d)
      : mu_(mu), B_(B), log_d_(log_d), dimension_(mu.size()), rank_(B.cols()) {
    static const char* function = "stan::variational::normal_lowrank";
    validate_mean(function, mu);
    validate_factor(function, B);
    validate_noise(function, log_d);
  }

  int dimension() const { return dimension_; }
  int rank() const { return rank_; }

  const Eigen::VectorXd& mean() const { return mu(); }
  const Eigen::VectorXd& mu() const { return mu_; }
  const Eigen::MatrixXd& B() const { return B_; }
  const Eigen::VectorXd& log_d() const { return log_d_; }

  void set_mu(const Eigen::VectorXd& mu) {
    static const char* function = "stan::variational::set_mu";
    validate_mean(function, mu);
    mu_ = mu;
  }

  void set_B(const Eigen::MatrixXd& B) {
    static const char* function = "stan::variational::set_B";
    validate_factor(function, B);
    B_ = B;
  }

  void set_log_d(const Eigen::VectorXd& log_d) {
    static const char* function = "stan::variational::set_log_d";
    validate_noise(function, log_d);
    log_d_ = log_d;
  }

  void set_to_zero() {
    mu_ = Eigen::VectorXd::Zero(dimension());
    B_ = Eigen::MatrixXd::Zero(dimension(), rank());
    log_d_ = Eigen::VectorXd::Zero(dimension());
  }

  normal_lowrank square() const {
    return normal_lowrank(Eigen::VectorXd(mu_.array().square()),
                          Eigen::MatrixXd(B_.array().square()),
                          Eigen::VectorXd(log_d_.array().square()));
  }

  normal_lowrank sqrt() const {
    return normal_lowrank(Eigen::VectorXd(mu_.array().sqrt()),
                          Eigen::MatrixXd(B_.array().sqrt()),
                          Eigen::VectorXd(log_d_.array().sqrt()));
  }

  normal_lowrank& operator=(const normal_lowrank& rhs) {
    static const char* function
        = "stan::variational::normal_lowrank::operator=";
    stan::math::check_size_match(function, "Dimension of lhs", dimension(),
                                 "Dimension of rhs", rhs.dimension());
    stan::math::check_size_match(function, "Rank of lhs", rank(), "Rank of rhs",
                                 rhs.rank());
    mu_ = rhs.mu();
    B_ = rhs.B();
    log_d_ = rhs.log_d();
    return *this;
  }

  normal_lowrank& operator+=(const normal_lowrank& rhs) {
    static const char* function
        = "stan::variational::normal_lowrank::operator+=";
    stan::math::check_size_match(function, "Dimension of lhs", dimension(),
                                 "Dimension of rhs", rhs.dimension());
    stan::math::check_size_match(function, "Rank of lhs", rank(), "Rank of rhs",
                                 rhs.rank());
    mu_ += rhs.mu();
    B_ += rhs.B();
    log_d_ += rhs.log_d();
    return *this;
  }

  inline normal_lowrank& operator/=(const normal_lowrank& rhs) {
    static const char* function
        = "stan::variational::normal_lowrank::operator/=";

    stan::math::check_size_match(function, "Dimension of lhs", dimension(),
                                 "Dimension of rhs", rhs.dimension());
    stan::math::check_size_match(function, "Rank of lhs", rank(), "Rank of rhs",
                                 rhs.rank());
    mu_.array() /= rhs.mu().array();
    B_.array() /= rhs.B().array();
    log_d_.array() /= rhs.log_d().array();
    return *this;
  }

  normal_lowrank& operator+=(double scalar) {
    mu_.array() += scalar;
    B_.array() += scalar;
    log_d_.array() += scalar;
    return *this;
  }

  normal_lowrank& operator*=(double scalar) {
    mu_ *= scalar;
    B_ *= scalar;
    log_d_ *= scalar;
    return *this;
  }

  double entropy() const {
    static int r = rank();
    static double mult = 0.5 * (1.0 + stan::math::LOG_TWO_PI);
    double result = mult * dimension();
    // Determinant by the matrix determinant lemma
    //   det(D^2 + B.B^T) = det(I + B^T.D^-2.B) * det(D^2)
    // where D^2 is diagonal and so can be computed accordingly
    result += 0.5
              * log((Eigen::MatrixXd::Identity(r, r)
                     + B_.transpose()
                           * log_d_.array()
                                 .exp()
                                 .square()
                                 .matrix()
                                 .asDiagonal()
                                 .inverse()
                           * B_)
                        .determinant());
    for (int d = 0; d < dimension(); ++d) {
      result += log_d_(d);
    }
    return result;
  }

  Eigen::VectorXd transform(const Eigen::VectorXd& eta) const {
    static const char* function
        = "stan::variational::normal_lowrank::transform";
    stan::math::check_size_match(function, "Dimension of input vector",
                                 eta.size(), "Sum of dimension and rank",
                                 dimension() + rank());
    stan::math::check_not_nan(function, "Input vector", eta);
    Eigen::VectorXd z = eta.head(rank());
    Eigen::VectorXd eps = eta.tail(dimension());
    return (log_d_.array().exp() * eps.array()).matrix() + (B_ * z) + mu_;
  }

  template <class BaseRNG>
  void sample(BaseRNG& rng, Eigen::VectorXd& eta) const {
    // Draw from standard normal and transform to real-coordinate space
    Eigen::VectorXd zeta = Eigen::VectorXd::Zero(dimension() + rank());
    for (int d = 0; d < dimension() + rank(); ++d)
      zeta(d) = stan::math::normal_rng(0, 1, rng);
    eta = transform(zeta);
  }

  /**
   * Draw a posterior sample from the normal distribution,
   * and return its log normal density. The constants are dropped.
   *
   * @param[in] rng Base random number generator.
   * @param[out] eta Vector to which the draw is assigned; dimension has to be
   * the same as the dimension of variational q. eta will be transformed into
   * variational posteriors.
   * @param[out] log_g The log  density in the variational approximation;
   * The constant term is dropped.
   * @throws std::range_error If the index is out of range.
   */
  template <class BaseRNG>
  void sample_log_g(BaseRNG& rng, Eigen::VectorXd& eta, double& log_g) const {
    // Draw from the approximation
    Eigen::VectorXd zeta = Eigen::VectorXd::Zero(dimension() + rank());
    for (int d = 0; d < dimension() + rank(); ++d) {
      zeta(d) = stan::math::normal_rng(0, 1, rng);
    }
    // Compute the log density before transformation
    log_g = calc_log_g(zeta);
    // Transform to real-coordinate space
    eta = transform(zeta);
  }

  template <class M, class BaseRNG>
  void calc_grad(normal_lowrank& elbo_grad, M& m, Eigen::VectorXd& cont_params,
                 int n_monte_carlo_grad, BaseRNG& rng,
                 callbacks::logger& logger) const {
    static const char* function
        = "stan::variational::normal_lowrank::calc_grad";

    stan::math::check_size_match(function, "Dimension of elbo_grad",
                                 elbo_grad.dimension(),
                                 "Dimension of variational q", dimension());
    stan::math::check_size_match(function, "Dimension of variational q",
                                 dimension(), "Dimension of variables in model",
                                 cont_params.size());

    stan::math::check_size_match(function, "Rank of elbo_grad",
                                 elbo_grad.rank(), "Rank of variational q",
                                 rank());

    Eigen::VectorXd mu_grad = Eigen::VectorXd::Zero(dimension());
    Eigen::MatrixXd B_grad = Eigen::MatrixXd::Zero(dimension(), rank());
    Eigen::VectorXd d_grad = Eigen::VectorXd::Zero(dimension());
    Eigen::VectorXd log_d_grad = Eigen::VectorXd::Zero(dimension());

    double tmp_lp = 0.0;
    Eigen::VectorXd eta = Eigen::VectorXd::Zero(dimension() + rank());
    Eigen::VectorXd zeta = Eigen::VectorXd::Zero(dimension());

    // (B.B^T + D^2)^-1 by Woodbury formula
    Eigen::MatrixXd d_inv2
        = log_d_.array().exp().square().matrix().asDiagonal().inverse();
    Eigen::MatrixXd Bt = B_.transpose();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(rank(), rank());
    Eigen::MatrixXd woodbury
        = d_inv2 - d_inv2 * B_ * (I + Bt * d_inv2 * B_).inverse() * Bt * d_inv2;

    Eigen::VectorXd tmp_mu_grad = Eigen::VectorXd::Zero(dimension());

    // Naive Monte Carlo integration
    static const int n_retries = 10;
    for (int i = 0, n_monte_carlo_drop = 0; i < n_monte_carlo_grad;) {
      // Draw from standard normal and transform to real-coordinate space
      for (int d = 0; d < dimension() + rank(); ++d) {
        eta(d) = stan::math::normal_rng(0, 1, rng);
      }
      Eigen::VectorXd z = eta.head(rank());
      Eigen::VectorXd eps = eta.tail(dimension());
      zeta = transform(eta);

      // (B.B^T + D^2)^-1 . (B.z + d*eps)
      Eigen::VectorXd woodbury_zeta = woodbury * (zeta - mu_);

      try {
        std::stringstream ss;
        stan::model::gradient(m, zeta, tmp_lp, tmp_mu_grad, &ss);
        if (ss.str().length() > 0)
          logger.info(ss);
        stan::math::check_finite(function, "Gradient of mu", mu_grad);

        mu_grad += tmp_mu_grad;
        for (int ii = 0; ii < dimension(); ++ii) {
          for (int jj = 0; jj <= ii && jj < rank(); ++jj) {
            B_grad(ii, jj) += (tmp_mu_grad(ii) + woodbury_zeta(ii)) * z(jj);
          }
          d_grad(ii) += (tmp_mu_grad(ii) + woodbury_zeta(ii)) * eps(ii);
          log_d_grad(ii) += d_grad(ii) * exp(log_d_(ii));
        }
        ++i;
      } catch (const std::exception& e) {
        ++n_monte_carlo_drop;
        if (n_monte_carlo_drop >= n_retries * n_monte_carlo_grad) {
          const char* name = "The number of dropped evaluations";
          const char* msg1 = "has reached its maximum amount (";
          int y = n_retries * n_monte_carlo_grad;
          const char* msg2
              = "). Your model may be either severely "
                "ill-conditioned or misspecified.";
          stan::math::domain_error(function, name, y, msg1, msg2);
        }
      }
    }
    mu_grad /= static_cast<double>(n_monte_carlo_grad);
    B_grad /= static_cast<double>(n_monte_carlo_grad);
    log_d_grad /= static_cast<double>(n_monte_carlo_grad);

    elbo_grad.set_mu(mu_grad);
    elbo_grad.set_B(B_grad);
    elbo_grad.set_log_d(log_d_grad);
  }

  double calc_log_g(const Eigen::VectorXd& eta) const {
    double log_g = 0;
    for (int d = 0; d < rank() + dimension(); ++d) {
      log_g += -stan::math::square(eta(d)) * 0.5;
    }
    return log_g;
  }
};

inline normal_lowrank operator+(normal_lowrank lhs, const normal_lowrank& rhs) {
  return lhs += rhs;
}

/**
 * Return a new approximation resulting from elementwise division of
 * of the first specified approximation by the second.
 *
 * @param[in] lhs First approximation.
 * @param[in] rhs Second approximation.
 * @return Elementwise division of the specified approximations.
 * @throw std::domain_error If the dimensionalities do not match.
 */
inline normal_lowrank operator/(normal_lowrank lhs, const normal_lowrank& rhs) {
  return lhs /= rhs;
}

/**
 * Return a new approximation resulting from elementwise addition
 * of the specified scalar to the mean and Cholesky factor of
 * covariance entries for the specified approximation.
 *
 * @param[in] scalar Scalar value
 * @param[in] rhs Approximation.
 * @return Addition of scalar to specified approximation.
 */
inline normal_lowrank operator+(double scalar, normal_lowrank rhs) {
  return rhs += scalar;
}

/**
 * Return a new approximation resulting from elementwise
 * multiplication of the specified scalar to the mean and Cholesky
 * factor of covariance entries for the specified approximation.
 *
 * @param[in] scalar Scalar value
 * @param[in] rhs Approximation.
 * @return Multiplication of scalar by the specified approximation.
 */
inline normal_lowrank operator*(double scalar, normal_lowrank rhs) {
  return rhs *= scalar;
}
}  // namespace variational
}  // namespace stan

#endif
