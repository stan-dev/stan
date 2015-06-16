#ifndef STAN_VARIATIONAL_ADVI_PARAMS_NORMAL_MEANFIELD__HPP
#define STAN_VARIATIONAL_ADVI_PARAMS_NORMAL_MEANFIELD__HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/scal/prob/normal_rng.hpp>

#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>

#include <stan/model/util.hpp>

#include <vector>
#include <ostream>

namespace stan {

  namespace variational {

    class advi_params_normal_meanfield {
    private:
      Eigen::VectorXd mu_;     // Mean vector
      Eigen::VectorXd omega_;  // Log standard deviation vector
      int dimension_;

    public:
      advi_params_normal_meanfield(const Eigen::VectorXd& mu,
                                   const Eigen::VectorXd& omega) :
      mu_(mu), omega_(omega), dimension_(mu.size()) {
        static const char* function =
          "stan::variational::advi_params_normal_meanfield";

        stan::math::check_size_match(function,
                             "Dimension of mean vector", dimension_,
                             "Dimension of log std vector", omega_.size() );
        for (int i = 0; i < dimension_; ++i) {
          stan::math::check_not_nan(function, "Mean vector", mu_(i));
          stan::math::check_not_nan(function, "Log std vector",
                                              omega_(i));
        }
      }

      // Accessors
      int dimension() const { return dimension_; }
      const Eigen::VectorXd& mu()    const { return mu_; }
      const Eigen::VectorXd& omega() const { return omega_; }

      // Mutators
      void set_mu(const Eigen::VectorXd& mu) {
        static const char* function =
          "stan::variational::advi_params_normal_meanfield::set_mu";

        stan::math::check_size_match(function,
                               "Dimension of input vector", mu.size(),
                               "Dimension of current vector", dimension_);
        stan::math::check_not_nan(function, "Input vector", mu);
        mu_ = mu;
      }

      void set_omega(const Eigen::VectorXd& omega) {
        static const char* function =
          "stan::variational::advi_params_normal_meanfield::set_omega";

        stan::math::check_size_match(function,
                               "Dimension of input vector", omega.size(),
                               "Dimension of current vector", dimension_);
        stan::math::check_not_nan(function, "Input vector", omega);
        omega_ = omega;
      }

      // Entropy of normal:
      // 0.5 * dim * (1+log2pi) + 0.5 * log det diag(sigma^2) =
      // 0.5 * dim * (1+log2pi) + sum(log(sigma)) =
      // 0.5 * dim * (1+log2pi) + sum(omega)
      double entropy() const {
        return 0.5 * static_cast<double>(dimension_) *
               (1.0 + stan::math::LOG_TWO_PI) + omega_.sum();
      }

      // Implement f^{-1}(\check{z}) = sigma * \check{z} + \mu
      Eigen::VectorXd transform(const Eigen::VectorXd& eta) const {
        static const char* function =
          "stan::variational::advi_params_normal_meanfield::transform";

        stan::math::check_size_match(function,
                         "Dimension of mean vector", dimension_,
                         "Dimension of input vector", eta.size() );
        stan::math::check_not_nan(function, "Input vector", eta);

        // exp(omega) * eta + mu
        return eta.array().cwiseProduct(omega_.array().exp()) + mu_.array();
      }

      /**
       * Draws samples from the variational distribution, which in this case is
       * a mean-field (diagonal) Gaussian.
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
       * MEAN-FIELD GRADIENTS
       *
       * Calculates the "blackbox" gradient with respect to BOTH the location
       * vector (mu) and the log-std vector (omega) in parallel.
       * It uses the same gradient computed from a set of Monte Carlo
       * samples.
       *
       * @tparam M                     class of model
       * @tparam BaseRNG               class of random number generator
       * @param  mu_grad               gradient of mean vector parameter
       * @param  omega_grad            gradient of log-std vector parameter
       * @param  cont_params           continuous parameters
       * @param  n_monte_carlo_grad    number of samples for gradient computation
       * @param  print_stream          stream for convergence assessment output
       */
      template <class M, class BaseRNG>
      void calc_grad(Eigen::VectorXd& mu_grad,
                     Eigen::VectorXd& omega_grad,
                     M& m,
                     Eigen::VectorXd& cont_params,
                     int n_monte_carlo_grad,
                     BaseRNG& rng,
                     std::ostream* print_stream) {
        static const char* function =
          "stan::variational::advi_params_normal_meanfield::calc_grad";

        stan::math::check_size_match(function,
                        "Dimension of mu grad vector", mu_grad.size(),
                        "Dimension of mean vector in variational q", dimension_);
        stan::math::check_size_match(function,
                        "Dimension of omega grad vector", omega_grad.size(),
                        "Dimension of mean vector in variational q", dimension_);
        stan::math::check_size_match(function,
                        "Dimension of muomega", dimension_,
                        "Dimension of variables in model", cont_params.size());

        // Initialize everything to zero
        mu_grad    = Eigen::VectorXd::Zero(dimension_);
        omega_grad = Eigen::VectorXd::Zero(dimension_);
        double tmp_lp = 0.0;
        Eigen::VectorXd tmp_mu_grad = Eigen::VectorXd::Zero(dimension_);
        Eigen::VectorXd eta  = Eigen::VectorXd::Zero(dimension_);
        Eigen::VectorXd zeta = Eigen::VectorXd::Zero(dimension_);

        // Naive Monte Carlo integration
        for (int i = 0; i < n_monte_carlo_grad; ++i) {
          // Draw from standard normal and transform to real-coordinate space
          for (int d = 0; d < dimension_; ++d) {
            eta(d) = stan::math::normal_rng(0, 1, rng);
          }
          zeta = transform(eta);

          stan::math::check_not_nan(function, "zeta", zeta);

          // Compute gradient step in real-coordinate space
          stan::model::gradient(m, zeta, tmp_lp, tmp_mu_grad,
                                print_stream);

          // Update mu
          mu_grad.array() = mu_grad.array() + tmp_mu_grad.array();

          // Update omega
          omega_grad.array() = omega_grad.array()
            + tmp_mu_grad.array().cwiseProduct(eta.array());
        }
        mu_grad    /= static_cast<double>(n_monte_carlo_grad);
        omega_grad /= static_cast<double>(n_monte_carlo_grad);

        // Multiply by exp(omega)
        omega_grad.array() =
          omega_grad.array().cwiseProduct(omega_.array().exp());

        // Add gradient of entropy term (just equal to element-wise 1 here)
        omega_grad.array() += 1.0;
      }
    };
  }  // variational
}  // stan

#endif
