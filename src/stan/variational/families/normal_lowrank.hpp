#ifndef STAN_VARIATIONAL_NORMAL_LOWRANK_HPP
#define STAN_VARIATIONAL_NORMAL_LOWRANK_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/math/prim/mat.hpp>
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
      Eigen::VectorXd d_;

      const int dimension_;
      const int rank_;

      void validate_mean(const char* function,
                         const Eigen::VectorXd& mu) {
        stan::math::check_not_nan(function, "Mean vector", mu);
        stan::math::check_size_match(function,
                                   "Dimension of input vector", mu.size(),
                                   "Dimension of current vector", dimension());
      }

      void validate_factor(const char* function,
                           const Eigen::MatrixXd& B) {
        stan::math::check_not_nan(function, "Low rank factor", B);
        stan::math::check_size_match(function,
                                     "Dimension of factor", B.rows(),
                                     "Dimension of approximation", dimension());
        stan::math::check_size_match(function,
                                     "Rank of factor", B.cols(),
                                     "Rank of approximation", rank());
      }

      void validate_noise(const char *function,
                          const Eigen::VectorXd& d) {
        stan::math::check_not_nan(function, "Noise vector", d);
        stan::math::check_size_match(function,
                                     "Dimension of noise vector", d.size(),
                                     "Dimension of approximation", dimension());
      }

    public:
      explicit normal_lowrank(size_t dimension, size_t rank)
      : mu_(Eigen::VectorXd::Zero(dimension)),
        B_(Eigen::MatrixXd::Zero(dimension, rank)),
        d_(Eigen::VectorXd::Zero(dimension)),
        dimension_(dimension),
        rank_(rank) {
      }

      explicit normal_lowrank(const Eigen::VectorXd& mu,
                              const Eigen::MatrixXd& B,
                              const Eigen::VectorXd& d)
      : mu_(mu), B_(B), d_(d), dimension_(mu.size()), rank_(B.cols()) {
        static const char* function = "stan::variational::normal_lowrank";
        validate_mean(function, mu);
        validate_factor(function, B);
        validate_noise(function, d);
      }

      int dimension() const { return dimension_; }
      int rank() const { return rank_; }

      const Eigen::VectorXd& mean() const { return mu(); }
      const Eigen::VectorXd& mu() const { return mu_; }
      const Eigen::MatrixXd& B() const { return B_; }
      const Eigen::VectorXd& d() const { return d_; }

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

      void set_d(const Eigen::VectorXd& d) {
        static const char* function = "stan::variational::set_d";
        validate_noise(function, d);
        d_ = d;
      }

      void set_to_zero() {
        mu_ = Eigen::VectorXd::Zero(dimension());
        B_ = Eigen::MatrixXd::Zero(dimension(), rank());
        d_ = Eigen::VectorXd::Zero(dimension());
      }

      double entropy() const {
        static int r = rank();
        static double mult = 0.5 * (1.0 + stan::math::LOG_TWO_PI);
        double result = mult * dimension();
        result += 0.5 * log((Eigen::MatrixXd::Identity(r, r) +
                          B_.transpose() *
                          d_.array().square().matrix().asDiagonal().inverse() *
                          B_).determinant());
        for (int d = 0; d < dimension(); ++d) {
          result += log(d_(d));
        }
        return result;
      }

      Eigen::VectorXd transform(const Eigen::VectorXd& eta) const {
        static const char* function =
          "stan::variational::normal_lowrank::transform";
        stan::math::check_size_match(function,
                             "Dimension of input vector", eta.size(),
                             "Sum of dimension and rank", dimension() + rank());
        stan::math::check_not_nan(function, "Input vector", eta);
        Eigen::VectorXd z = eta.head(rank());
        Eigen::VectorXd eps = eta.tail(dimension());
        return (d_.cwiseProduct(eps)) + (B_ * z) + mu_;
      }

      template <class M, class BaseRNG>
      void calc_grad(normal_lowrank& elbo_grad,
                     M& m,
                     Eigen::VectorXd& cont_params,
                     int n_monte_carlo_grad,
                     BaseRNG& rng,
                     callbacks::logger& logger) const {
        static const char* function =
          "stan::variational::normal_lowrank::calc_grad";

        Eigen::VectorXd mu_grad = Eigen::VectorXd::Zero(dimension());
        Eigen::MatrixXd B_grad = Eigen::MatrixXd::Zero(dimension(), rank());
        Eigen::VectorXd d_grad = Eigen::VectorXd::Zero(dimension());

        double tmp_lp = 0.0;
        Eigen::VectorXd tmp_mu_grad = Eigen::VectorXd::Zero(dimension());
        Eigen::VectorXd eta = Eigen::VectorXd::Zero(dimension() + rank());
        Eigen::VectorXd zeta = Eigen::VectorXd::Zero(dimension());

        Eigen::MatrixXd inv_noise =
          d_.array().square().matrix().asDiagonal().inverse();
        Eigen::VectorXd var_grad_left = inv_noise - inv_noise * B_ *
                                    (Eigen::MatrixXd::Identity(rank(), rank()) +
                                    B_.transpose() * inv_noise * B_).inverse()
                                    * B_.transpose() * inv_noise;
        Eigen::VectorXd tmp_var_grad_left = Eigen::VectorXd::Zero(dimension());

        // Naive Monte Carlo integration
        static const int n_retries = 10;
        for (int i = 0, n_monte_carlo_drop = 0; i < n_monte_carlo_grad; ) {
          // Draw from standard normal and transform to real-coordinate space
          for (int d = 0; d < dimension() + rank(); ++d) {
            eta(d) = stan::math::normal_rng(0, 1, rng);
          }
          Eigen::VectorXd z = eta.head(rank());
          Eigen::VectorXd eps = eta.tail(dimension());
          zeta = transform(eta);
          try {
            std::stringstream ss;
            stan::model::gradient(m, zeta, tmp_lp, tmp_mu_grad, &ss);
            if (ss.str().length() > 0)
              logger.info(ss);
            stan::math::check_finite(function, "Gradient of mu", tmp_mu_grad);

            mu_grad += tmp_mu_grad;
            tmp_var_grad_left = tmp_mu_grad + var_grad_left * (zeta - mu);
            for (int ii = 0; ii < dimension(); ++ii) {
              for (int jj = 0; jj <= ii && jj < rank(); ++jj) {
                B_grad(ii, jj) += tmp_var_grad_left(ii) * z(jj);
              }
              d_grad(ii) += tmp_var_grad_left(ii) * eps(ii);
            }
            ++i;
          } catch (const std::exception& e) {
            ++n_monte_carlo_drop;
            if (n_monte_carlo_drop >= n_retries * n_monte_carlo_grad) {
              const char* name = "The number of dropped evaluations";
              const char* msg1 = "has reached its maximum amount (";
              int y = n_retries * n_monte_carlo_grad;
              const char* msg2 = "). Your model may be either severely "
                "ill-conditioned or misspecified.";
              stan::math::domain_error(function, name, y, msg1, msg2);
            }
          }
        }
        mu_grad /= static_cast<double>(n_monte_carlo_grad);
        B_grad /= static_cast<double>(n_monte_carlo_grad);
        d_grad /= static_cast<double>(n_monte_carlo_grad);

        elbo_grad.set_mu(mu_grad);
        elbo_grad.set_B(B_grad);
        elbo_grad.set_d(d_grad);
      }

      double calc_log_g(const Eigen::VectorXd& eta) const {
        double log_g = 0;
        for (int d = 0; d < rank() + dimension(); ++d) {
          log_g += -stan::math::square(eta(d)) * 0.5;
        }
        return log_g;
      }
    };
  }
}

#endif

