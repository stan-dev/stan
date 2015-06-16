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
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>

#include <stan/model/util.hpp>

#include <vector>
#include <ostream>

namespace stan {

  namespace variational {

    /*
     * MULTIVARIATE NORMAL DISTRIBUTION FULL-RANK
     *
     * Variational family as full-rank multivariate normal distribution, with free
     * parameters mean and cholesky factor of the covariance
     *
     * @param  mu     mean vector
     * @param  L_chol cholesky factor of covariance (\Sigma = L_chol *
     *                L_chol.transpose())
     */
    class normal_fullrank {
    private:
      Eigen::VectorXd mu_;
      Eigen::MatrixXd L_chol_;
      int dimension_;

    public:
      normal_fullrank(const Eigen::VectorXd& mu,
                                  const Eigen::MatrixXd& L_chol) :
      mu_(mu), L_chol_(L_chol), dimension_(mu.size()) {
        static const char* function =
          "stan::variational::normal_fullrank";

        stan::math::check_size_match(function,
                               "Dimension of mean vector",     dimension_,
                               "Dimension of Cholesky factor", L_chol_.rows() );
        stan::math::check_not_nan(function, "Mean vector", mu_);
        stan::math::check_lower_triangular(function,
                               "Cholesky factor", L_chol_);
      }

      // Accessors
      int dimension() const { return dimension_; }
      const Eigen::VectorXd& mu()     const { return mu_; }
      const Eigen::MatrixXd& L_chol() const { return L_chol_; }

      // Mutators
      void set_mu(const Eigen::VectorXd& mu) {
        static const char* function =
          "stan::variational::normal_fullrank::set_mu";

        stan::math::check_size_match(function,
                               "Dimension of input vector", mu.size(),
                               "Dimension of current vector", dimension_);
        stan::math::check_not_nan(function, "Input vector", mu);
        mu_ = mu;
      }

      void set_L_chol(const Eigen::MatrixXd& L_chol) {
        static const char* function =
          "stan::variational::normal_fullrank::set_L_chol";

        stan::math::check_size_match(function,
                               "Dimension of mean vector",     dimension_,
                               "Dimension of Cholesky factor", L_chol.rows());
        stan::math::check_lower_triangular(function,
                               "Cholesky factor", L_chol);
        L_chol_ = L_chol;
      }

      // Entropy of normal:
      // 0.5 * dim * (1+log2pi) + 0.5 * log det (L^T L) =
      // 0.5 * dim * (1+log2pi) + sum(log(abs(diag(L))))
      double entropy() const {
        double tmp(0.0);
        double result(
          0.5 * static_cast<double>(dimension_) * (1.0 + stan::math::LOG_TWO_PI));
        for (int d = 0; d < dimension_; ++d) {
          tmp = fabs(L_chol_(d, d));
          if (tmp != 0.0) {
            result += log(tmp);
          }
        }
        return result;
      }

      // Implements S^{-1}(eta) = L*eta + \mu
      Eigen::VectorXd
      transform(const Eigen::VectorXd& eta) const {
        static const char* function =
          "stan::variational::normal_fullrank::transform";

        stan::math::check_size_match(function,
                         "Dimension of input vector", eta.size(),
                         "Dimension of mean vector",  dimension_);
        stan::math::check_not_nan(function, "Input vector", eta);

        return (L_chol_*eta).array() + mu_.array();
      }

      /**
       * Draws samples from the variational distribution, which in this case is
       * a fullrank Gaussian.
       *
       * @tparam BaseRNG           class of random number generator
       * @return                   a sample from the variational distribution
       */
      template <class BaseRNG>
      Eigen::VectorXd sample(BaseRNG& rng) const {
        Eigen::VectorXd eta = Eigen::VectorXd::Zero(dimension_);

        // Draw from standard normal and transform to real-coordinate space
        for (int d = 0; d < dimension_; ++d) {
          eta(d) = stan::math::normal_rng(0, 1, rng);
        }

        return transform(eta);
      }

      /**
       * FULL-RANK GRADIENTS
       *
       * Calculates the "blackbox" gradient with respect to BOTH the location
       * vector (mu) and the cholesky factor of the scale matrix (L) in
       * parallel. It uses the same gradient computed from a set of Monte Carlo
       * samples
       *
       * @tparam M                     class of model
       * @tparam BaseRNG               class of random number generator
       * @param  mu_grad               gradient of location vector parameter
       * @param  L_grad                gradient of scale matrix parameter
       * @param  cont_params           continuous parameters
       * @param  n_monte_carlo_grad    number of samples for gradient computation
       * @param  print_stream          stream for convergence assessment output
       */
      template <class M, class BaseRNG>
      void calc_grad(Eigen::VectorXd& mu_grad,
                     Eigen::MatrixXd& L_grad,
                     M& m,
                     Eigen::VectorXd& cont_params,
                     int n_monte_carlo_grad,
                     BaseRNG& rng,
                     std::ostream* print_stream) {
        static const char* function =
          "stan::variational::normal_fullrank::calc_grad";

        stan::math::check_size_match(function,
                        "Dimension of muL", dimension_,
                        "Dimension of variables in model", cont_params.size());
        stan::math::check_size_match(function,
                        "Dimension of mu grad vector", mu_grad.size(),
                        "Dimension of mean vector in variational q", dimension_);
        stan::math::check_square(function, "Scale matrix", L_grad);
        stan::math::check_size_match(function,
                        "Dimension of scale matrix", L_grad.rows(),
                        "Dimension of mean vector in variational q", dimension_);

        // Initialize everything to zero
        mu_grad = Eigen::VectorXd::Zero(dimension_);
        L_grad  = Eigen::MatrixXd::Zero(dimension_, dimension_);
        double tmp_lp = 0.0;
        Eigen::VectorXd tmp_mu_grad = Eigen::VectorXd::Zero(dimension_);
        Eigen::VectorXd eta = Eigen::VectorXd::Zero(dimension_);
        Eigen::VectorXd zeta = Eigen::VectorXd::Zero(dimension_);

        // Naive Monte Carlo integration
        for (int i = 0; i < n_monte_carlo_grad; ++i) {
          // Draw from standard normal and transform to real-coordinate space
          for (int d = 0; d < dimension_; ++d) {
            eta(d) = stan::math::normal_rng(0, 1, rng);
          }
          zeta = transform(eta);

          // Compute gradient step in real-coordinate space
          stan::model::gradient(m, zeta, tmp_lp, tmp_mu_grad,
                                print_stream);

          // Update mu
          mu_grad += tmp_mu_grad;

          // Update L (lower triangular)
          for (int ii = 0; ii < dimension_; ++ii) {
            for (int jj = 0; jj <= ii; ++jj) {
              L_grad(ii, jj) += tmp_mu_grad(ii) * eta(jj);
            }
          }
        }
        mu_grad /= static_cast<double>(n_monte_carlo_grad);
        L_grad  /= static_cast<double>(n_monte_carlo_grad);

        // Add gradient of entropy term
        L_grad.diagonal().array() += L_chol_.diagonal().array().inverse();
      }
    };
  }  // variational
}  // stan

#endif
