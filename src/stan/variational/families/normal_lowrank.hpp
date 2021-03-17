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
/**
 * Variational family approximation with low-rank multivariate normal
 * distribution.
 */
class normal_lowrank : public base_family {
 private:
  /**
   * Mean vector.
   */
  Eigen::VectorXd mu_;

  /**
   * Low-rank covariance factor.
   */
  Eigen::MatrixXd B_;

  /**
   * Additive white noise for covariance positive definite-ness.
   */
  Eigen::VectorXd log_d_;

  /**
   * Dimensionality of the distribution.
   */
  const int dimension_;

  /**
   * Rank of the low-rank covariance factor.
   */
  const int rank_;

  /**
   * Raise a domain exception if the specified vector contains not-a-number
   * values or is not of the correct dimension.
   *
   * @param[in] mu Mean vector.
   * @throw std::domain_error If the mean vector contains NaN values or does
   * not match this distribution's dimensionality.
   */
  void validate_mean(const char* function, const Eigen::VectorXd& mu) {
    stan::math::check_not_nan(function, "Mean vector", mu);
    stan::math::check_size_match(function, "Dimension of input vector",
                                 mu.size(), "Dimension of current vector",
                                 dimension());
  }

  /**
   * Raise a domain exception if the specified matrix is not of the correct
   * dimension or rank, is not lower triangular, or contains not-a-number
   * values.
   *
   * @param[in] B Low-rank factor.
   * @throw std::domain_error If the matrix is not of the correct dimension or
   * rank, or if it contains not-a-number values.
   */
  void validate_factor(const char* function, const Eigen::MatrixXd& B) {
    stan::math::check_not_nan(function, "Low rank factor", B);
    stan::math::check_lower_triangular(function, "Low rank factor", B);
    stan::math::check_size_match(function, "Dimension of mean vector",
                                 dimension(), "Dimension of low-rank factor",
                                 B.rows());
    stan::math::check_size_match(function, "Rank of factor", B.cols(),
                                 "Rank of approximation", rank());
  }

  /**
   * Raise a domain exception if the specified matrix is not of the correct
   * dimension, or contains not-a-number-values.
   *
   * @param[in] log_d Log std vector.
   * @throw std::domain_error If the matrix is not of the correct dimension or
   * rank, or if it contains not-a-number values.
   */
  void validate_noise(const char* function, const Eigen::VectorXd& log_d) {
    stan::math::check_not_nan(function, "log std vector", log_d);
    stan::math::check_size_match(function, "Dimension of mean vector",
                                 dimension(), "Dimension of log std vector",
                                 log_d.size());
  }

 public:
  /**
   * Construct a variational distribution of the specified dimensionality with
   * a zero mean, zero low-rank factor matrix, and a zero log-std vector,
   * corresponding to an identity covariance.
   *
   * @param[in] dimension The dimensionality of the distribution.
   * @param[in] rank The rank of the low-rank factor.
   */
  explicit normal_lowrank(size_t dimension, size_t rank)
      : mu_(Eigen::VectorXd::Zero(dimension)),
        B_(Eigen::MatrixXd::Zero(dimension, rank)),
        log_d_(Eigen::VectorXd::Zero(dimension)),
        dimension_(dimension),
        rank_(rank) {}

  /**
   * Construct a variational distribution with specified mean vector, and
   * identity covariance.
   *
   * @params[in] mu Mean vector.
   * @params[in] rank Rank of the approximation.
   */
  explicit normal_lowrank(const Eigen::VectorXd& mu, size_t rank)
      : mu_(mu),
        B_(Eigen::MatrixXd::Zero(mu.size(), rank)),
        log_d_(Eigen::VectorXd::Zero(mu.size())),
        dimension_(mu.size()),
        rank_(rank) {}

  /**
   * Construct a variational distribution with specified mean and covariance
   * parameters.
   *
   * @param[in] mu Mean vector.
   * @param[in] B Low-rank covariance factor.
   * @param[in] log_d Additive log std vector.
   * @throws std::domain_error If the low-rank covariance factor is not not
   * lower-triangular, if the parameters have inconsistent dimensionality,
   * or if any NaNs are present.
   */
  explicit normal_lowrank(const Eigen::VectorXd& mu, const Eigen::MatrixXd& B,
                          const Eigen::VectorXd& log_d)
      : mu_(mu), B_(B), log_d_(log_d), dimension_(mu.size()), rank_(B.cols()) {
    static const char* function = "stan::variational::normal_lowrank";
    validate_mean(function, mu);
    validate_factor(function, B);
    validate_noise(function, log_d);
  }

  /**
   * Return the dimensionality of the approximation.
   */
  int dimension() const { return dimension_; }

  /**
   * Return the rank of the approximation.
   */
  int rank() const { return rank_; }

  /**
   * Return the mean vector.
   */
  const Eigen::VectorXd& mean() const { return mu(); }

  /**
   * Return the mean vector.
   */
  const Eigen::VectorXd& mu() const { return mu_; }

  /**
   * Return the low-rank covariance factor.
   */
  const Eigen::MatrixXd& B() const { return B_; }

  /**
   * Return the additive log-std.
   */
  const Eigen::VectorXd& log_d() const { return log_d_; }

  /**
   * Set the mean vector.
   *
   * @param[in] mu Mean vector.
   * @throw std::domain_error If the size of the specified mean vector does not
   * match the stored dimension of this approximation.
   */
  void set_mu(const Eigen::VectorXd& mu) {
    static const char* function = "stan::variational::set_mu";
    validate_mean(function, mu);
    mu_ = mu;
  }

  /**
   * Set the low-rank factor to the specified value.
   *
   * @param[in] B The low-rank factor of the covariance matrix.
   * @throw std::domain_error If the specified matrix does not have the correct
   * dimension or rank, is not lower triangular, or contains NaNs.
   */
  void set_B(const Eigen::MatrixXd& B) {
    static const char* function = "stan::variational::set_B";
    validate_factor(function, B);
    B_ = B;
  }

  /**
   * Set the additive log-std component of the covariance.
   *
   * @param[in] log_d The log-std vector.
   * @throw std::domain_error If the size of the log-std vector does not match
   * the dimension of the approximation, or contains NaNs.
   */
  void set_log_d(const Eigen::VectorXd& log_d) {
    static const char* function = "stan::variational::set_log_d";
    validate_noise(function, log_d);
    log_d_ = log_d;
  }

  /**
   * Set the mean, low-rank factor, and log-std of the approximation to zero.
   */
  void set_to_zero() {
    mu_ = Eigen::VectorXd::Zero(dimension());
    B_ = Eigen::MatrixXd::Zero(dimension(), rank());
    log_d_ = Eigen::VectorXd::Zero(dimension());
  }

  /**
   * Return a new low-rank approximation resulting from squaring the entries in
   * the mean, low-rank factor, and log-std vector for the covariance matrix.
   * The new approximation does not hold any references to this approximation.
   */
  normal_lowrank square() const {
    return normal_lowrank(Eigen::VectorXd(mu_.array().square()),
                          Eigen::MatrixXd(B_.array().square()),
                          Eigen::VectorXd(log_d_.array().square()));
  }

  /**
   * Return a new low-rank approximation resulting from taking the square root
   * of the entries in the mean, low-rank factor, and log-std vector for the
   * covariance matrix. The new approximation does not hold any references to
   * this approximation.
   */
  normal_lowrank sqrt() const {
    return normal_lowrank(Eigen::VectorXd(mu_.array().sqrt()),
                          Eigen::MatrixXd(B_.array().sqrt()),
                          Eigen::VectorXd(log_d_.array().sqrt()));
  }

  /**
   * Return this approximation after setting its mean vector, low-rank factor,
   * and log-std vector to the values given by the specified approximation.
   *
   * @param[in] rhf Approximation from which to gather the mean and covariance
   * parameters.
   * @return This approximation after assignment.
   * @throw std::domain_error If the dimensionality or rank of the specified
   * approximation does not match this approximation's dimensionality or rank.
   */
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

  /**
   * Add the mean, low-rank covariance factor, and log-std of the specified
   * approximation to this approximation.
   *
   * @param[in] rhs Approximation from which to gather the mean and covariance
   * parameters.
   * @return This approximation after adding the specified approximation.
   * @throw std::domain_error If the dimensionality or rank of the specified
   * approximation does not match this approximation's dimensionality.
   */
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

  /**
   * Return this approximation after elementwise division by the specified
   * approximation's mean, low-rank factor, and log-std vector.
   *
   * @param[in] rhs Approximation from which to gather the mean and covariance
   * parameters.
   * @return This approximation after dividing by the specified approximation.
   * @throw std::domain_error If the dimensionality or rank of the specified
   * approximation does not match this approximation's dimensionality.
   */
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

  /**
   * Return this approximation after adding the specified scalar to each entry
   * in the mean, low-rank covariance factor, and log-std vector.
   *
   * <b>Warning:</b> No finiteness check is made on the scalar, so it may
   * introduce NaNs.
   *
   * @param[in] scalar Scalar to add.
   * @return This approximation after elementwise addition of the specified
   * scalar.
   */
  normal_lowrank& operator+=(double scalar) {
    mu_.array() += scalar;
    B_.array() += scalar;
    log_d_.array() += scalar;
    return *this;
  }

  /**
   * Return this approximation after multiplying the mean, low-rank factor, and
   * log-std vector by the specified scalar.
   *
   * <b>Warning:</b> No finiteness check is made on the scalar, so it may
   * introduce NaNs.
   *
   * @param[in] scalar Scalar to multiply by.
   * @return This approximation after elementwise multiplication of the
   * specified scalar.
   */
  normal_lowrank& operator*=(double scalar) {
    mu_ *= scalar;
    B_ *= scalar;
    log_d_ *= scalar;
    return *this;
  }

  /**
   * Return the entropy of this approximation.
   *
   * <p>The entropy is defined using the matrix determinant lemma as
   *   0.5 * dim * (1 + log2pi) + 0.5 * log det (B^T B + D^2)
   * where
   *   log det (B^T B + D^2) = log det(I + B^T D^2 B) + 2 * log det(D)
   *                         = log det(I + B^T D^2 B) + 2 * sum(log_d)
   * where the last equality holds because the representation of D stored is
   * its diag vector.
   *
   * @return Entropy of this approximation.
   */
  double entropy() const {
    static int r = rank();
    static double mult = 0.5 * (1.0 + stan::math::LOG_TWO_PI);
    double result = mult * dimension();
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

  /**
   * Return the transform of the specified vector using the mean and covariance
   * parameters.
   *
   * The transform is defined by
   * S^{-1}(eta) = B * z + D * eps + mu
   * where D = diag(exp(log_d)), z = head(eta, rank) and
   * eps = tail(eta, dimension).
   */
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

  /**
   * Sample from the variational distribution.
   *
   * @tparam BaseRNG Random number generator class.
   * @param[in] rng Random number generator.
   * @param[out] zeta Output variational standard normals.
   */
  template <class BaseRNG>
  void sample(BaseRNG& rng, Eigen::VectorXd& zeta) const {
    // Draw from standard normal and transform to real-coordinate space
    Eigen::VectorXd eta = Eigen::VectorXd::Zero(dimension() + rank());
    for (int d = 0; d < dimension() + rank(); ++d)
      eta(d) = stan::math::normal_rng(0, 1, rng);
    zeta = transform(eta);
  }

  /**
   * Sample from the variational distribution and return its log normal
   * density.
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

  /**
   * Calculates the gradient with respect to the mean and covariance parameters
   * in parallel.
   *
   * @tparam M Model class.
   * @tparam BaseRNG Class of base random number generator.
   * @param[in] elbo_grad Approximation to store gradient.
   * @param[in] m Model.
   * @param[in] cont_params Continuous parameters.
   * @param[in] n_monte_carlo_grad Sample size for gradient computation.
   * @param[in,out] rng Random number generator.
   * @param[in,out] logger Logger for messages.
   * @throw std::domain_error If the number of divergent iterations exceeds its
   * specified bounds.
   */
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

  /**
   * Calculate the log-density of a vector of independent std normals.
   *
   * @param[in] eta The random sample.
   * @return The log-probability of the random sample.
   */
  double calc_log_g(const Eigen::VectorXd& eta) const {
    double log_g = 0;
    for (int d = 0; d < rank() + dimension(); ++d) {
      log_g += -stan::math::square(eta(d)) * 0.5;
    }
    return log_g;
  }
};

/**
 * Return a new approximation resulting from elementwise addition of
 * of the first specified approximation to the second.
 *
 * @param[in] lhs First approximation.
 * @param[in] rhs Second approximation.
 * @return Elementwise addition of the specified approximations.
 * @throw std::domain_error If the dimensionalities do not match.
 */
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
 * of the specified scalar to the mean, low-rank factor, and log-std
 * vector entries for the specified approximation.
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
 * multiplication of the specified scalar to the mean low-rank factor,
 * and log-std for the specified approximation.
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
