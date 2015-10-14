#ifndef STAN_VARIATIONAL_NORMAL_FULLRANK_HPP
#define STAN_VARIATIONAL_NORMAL_FULLRANK_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/LDLT_factor.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/scal/prob/normal_rng.hpp>

#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/err/check_lower_triangular.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/err/domain_error.hpp>

#include <stan/variational/base_family.hpp>

#include <stan/model/util.hpp>
#include <algorithm>
#include <ostream>
#include <vector>

namespace stan {

  namespace variational {

    class normal_fullrank : public base_family {
    public:
      // TODO explicitly describe why all these cosntructors
      explicit normal_fullrank(size_t dimension) :
        dimension_(dimension) {
        mu_ = Eigen::VectorXd::Zero(dimension_);
        L_  = Eigen::MatrixXd::Identity(dimension_, dimension_);
      }

      explicit normal_fullrank(const Eigen::VectorXd& cont_params) :
        dimension_(cont_params.size()) {
        set_mu(cont_params);
        L_ = Eigen::MatrixXd::Identity(dimension_, dimension_);
      }

      normal_fullrank(const Eigen::VectorXd& mu,
                      const Eigen::MatrixXd& L) :
        dimension_(mu.size()) {
        set_mu(mu);
        set_L(L);
      }

      int dimension() const { return dimension_; }

      normal_fullrank square() const {
        return normal_fullrank(Eigen::VectorXd(mu_.array().square()),
                               Eigen::MatrixXd(L_.array().square()));
      }

      normal_fullrank sqrt() const {
        return normal_fullrank(Eigen::VectorXd(mu_.array().sqrt()),
                               Eigen::MatrixXd(L_.array().sqrt()));
      }

      normal_fullrank operator=(const normal_fullrank& rhs) {
        static const char* function =
          "stan::variational::normal_fullrank::operator=";

        stan::math::check_size_match(function,
                             "Dimension of lhs", dimension_,
                             "Dimension of rhs", rhs.dimension());

        mu_ = rhs.mu();
        L_ = rhs.L();
        return *this;
      }

      normal_fullrank operator+=(const normal_fullrank& rhs) {
        static const char* function =
          "stan::variational::normal_fullrank::operator+=";

        stan::math::check_size_match(function,
                             "Dimension of lhs", dimension_,
                             "Dimension of rhs", rhs.dimension());

        mu_ += rhs.mu();
        L_ += rhs.L();
        return *this;
      }

      normal_fullrank operator/=(const normal_fullrank& rhs) {
        static const char* function =
          "stan::variational::normal_fullrank::operator/=";

        stan::math::check_size_match(function,
                             "Dimension of lhs", dimension_,
                             "Dimension of rhs", rhs.dimension());

        mu_.array() /= rhs.mu().array();
        L_.array() /= rhs.L().array();
        return *this;
      }

      normal_fullrank operator+=(double x) {
        mu_.array() += x;
        L_.array() += x;
        return *this;
      }

      normal_fullrank operator*=(double x) {
        mu_ *= x;
        L_ *= x;
        return *this;
      }

      const Eigen::VectorXd& mean() const {
        return mu();
      }

      // 0.5 * dim * (1+log2pi) + 0.5 * log det (L^T L) =
      // 0.5 * dim * (1+log2pi) + sum(log(abs(diag(L))))
      // TODO entropy, zeros, various checks
      double entropy() const {
        double tmp = 0.0;
        double result = 0.5 * dimension_ * (1.0 + stan::math::LOG_TWO_PI);
        for (int d = 0; d < dimension_; ++d) {
          tmp = fabs(L_(d, d));
          if (tmp != 0.0) {
            result += log(tmp);
          }
        }
        return result;
      }

      // Implements S^{-1}(eta) = L*eta + \mu
      Eigen::VectorXd transform(const Eigen::VectorXd& eta) const {
        static const char* function =
          "stan::variational::normal_fullrank::transform";

        stan::math::check_size_match(function,
                         "Dimension of input vector", eta.size(),
                         "Dimension of mean vector",  dimension_);
        stan::math::check_not_nan(function, "Input vector", eta);

        return (L_ * eta) + mu_;
      }

      template <class BaseRNG>
      void sample(BaseRNG& rng, Eigen::VectorXd& eta) const {
        // Draw from standard normal and transform to real-coordinate space
        for (int d = 0; d < dimension_; ++d) {
          eta(d) = stan::math::normal_rng(0, 1, rng);
        }

        eta = transform(eta);
      }

      /**
       * Calculates the "black box" gradient with respect to the location
       * vector (mu) and the cholesky factor of the scale matrix (L).
       *
       * @param[out] elbo_grad     parameters to store gradient
       * @param m                  model
       * @param cont_params        continuous parameters
       * @param n_monte_carlo_grad number of samples for gradient computation
       * @param rng                random number generator
       * @param print_stream       stream for convergence assessment output
       */
      template <class M, class BaseRNG>
      void calc_grad(normal_fullrank& elbo_grad,
                     M& m,
                     Eigen::VectorXd& cont_params,
                     int n_monte_carlo_grad,
                     BaseRNG& rng,
                     std::ostream* print_stream) const {
        static const char* function =
          "stan::variational::normal_fullrank::calc_grad";

        stan::math::check_size_match(function,
                        "Dimension of elbo_grad", elbo_grad.dimension(),
                        "Dimension of variational q", dimension_);
        stan::math::check_size_match(function,
                        "Dimension of variational q", dimension_,
                        "Dimension of variables in model", cont_params.size());

        Eigen::VectorXd mu_grad = Eigen::VectorXd::Zero(dimension_);
        Eigen::MatrixXd L_grad  = Eigen::MatrixXd::Zero(dimension_, dimension_);
        double tmp_lp(0.0);
        Eigen::VectorXd tmp_mu_grad = Eigen::VectorXd::Zero(dimension_);
        Eigen::VectorXd eta         = Eigen::VectorXd::Zero(dimension_);
        Eigen::VectorXd zeta        = Eigen::VectorXd::Zero(dimension_);
        // TODO
        //Eigen::VectorXd eta(dimension_);
        //Eigen::VectorXd zeta(dimension_);

        // Monte Carlo integration
        int i = 0;
        int n_monte_carlo_drop = 0;
        while (i < n_monte_carlo_grad) {
          for (int d = 0; d < dimension_; ++d) {
            eta(d) = stan::math::normal_rng(0, 1, rng);
          }
          zeta = transform(eta);

          try {
            stan::model::gradient(m, zeta, tmp_lp, tmp_mu_grad, print_stream);
            stan::math::check_finite(function, "Gradient of mu", tmp_mu_grad);

            mu_grad += tmp_mu_grad;
            for (int ii = 0; ii < dimension_; ++ii) {
              for (int jj = 0; jj <= ii; ++jj) {
                L_grad(ii, jj) += tmp_mu_grad(ii) * eta(jj);
              }
            }
            i += 1;
          } catch (std::exception& e) {
            this->write_error_msg_(print_stream, e);
            n_monte_carlo_drop += 1;
            if (n_monte_carlo_drop >= 5*n_monte_carlo_grad) {
              const char* name = "The number of dropped evaluations";
              const char* msg1 = "has reached its maximum amount (";
              int y = 5*n_monte_carlo_grad;
              const char* msg2 = "). Your model may be either severely "
                "ill-conditioned or misspecified.";
              stan::math::domain_error(function, name, y, msg1, msg2);
            }
          }
        }
        mu_grad /= n_monte_carlo_grad;
        L_grad  /= n_monte_carlo_grad;

        // Add gradient of entropy
        L_grad.diagonal().array() += L_.diagonal().array().inverse();

        // Set parameters to argument
        elbo_grad.set_mu(mu_grad);
        elbo_grad.set_L(L_grad);
      }

    private:
      Eigen::VectorXd mu_;
      Eigen::MatrixXd L_;
      int dimension_;

      const Eigen::VectorXd& mu() const { return mu_; }
      const Eigen::MatrixXd& L()  const { return L_; }

      // TODO @throws
      void set_mu(const Eigen::VectorXd& mu) {
        static const char* function =
          "stan::variational::normal_fullrank::set_mu";

        stan::math::check_size_match(function,
                               "Dimension of input vector", mu.size(),
                               "Dimension of current vector", dimension_);
        stan::math::check_not_nan(function, "Input vector", mu);
        mu_ = mu;
      }

      void set_L(const Eigen::MatrixXd& L) {
        static const char* function =
          "stan::variational::normal_fullrank::set_L";

        stan::math::check_square(function, "Input matrix", L);
        stan::math::check_lower_triangular(function,
                               "Input matrix", L);
        stan::math::check_size_match(function,
                               "Dimension of input matrix", L.rows(),
                               "Dimension of current matrix", dimension_);
        stan::math::check_not_nan(function, "Input matrix", L);
        L_ = L;
      }
    };

    normal_fullrank operator+(normal_fullrank lhs, const normal_fullrank& rhs) { return lhs += rhs; }
    normal_fullrank operator/(normal_fullrank lhs, const normal_fullrank& rhs) { return lhs /= rhs; }
    normal_fullrank operator+(double x, normal_fullrank rhs) { return rhs += x; }
    normal_fullrank operator*(double x, normal_fullrank rhs) { return rhs *= x; }
  }  // variational
}  // stan

#endif
