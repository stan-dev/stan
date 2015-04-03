#ifndef STAN__VARIATIONAL__ADVI_PARAMS_MEANFIELD__HPP
#define STAN__VARIATIONAL__ADVI_PARAMS_MEANFIELD__HPP

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/fun/max.hpp>

namespace stan {

  namespace variational {

    class advi_params_meanfield {

    private:

      Eigen::VectorXd mu_;          // Mean vector
      Eigen::VectorXd sigma_tilde_; // Log standard deviation vector
      int dimension_;

    public:

      advi_params_meanfield(const Eigen::VectorXd& mu,
                            const Eigen::VectorXd& sigma_tilde) :
      mu_(mu),
      sigma_tilde_(sigma_tilde),
      dimension_(mu.size()) {
        static const char* function =
          "stan::variational::advi_params_meanfield";

        stan::math::check_size_match(function,
                               "Dimension of mean vector", dimension_,
                               "Dimension of std vector", sigma_tilde_.size() );
      };

      virtual ~advi_params_meanfield() {}; // No-op

      // Accessors
      int dimension() const { return dimension_; }
      const Eigen::VectorXd& mu()          const { return mu_; }
      const Eigen::VectorXd& sigma_tilde() const { return sigma_tilde_; }

      // Mutators
      void set_mu(const Eigen::VectorXd& mu) {
        static const char* function =
          "stan::variational::advi_params_meanfield::set_mu";

        stan::math::check_size_match(function,
                               "Dimension of input vector", mu.size(),
                               "Dimension of current vector", dimension_ );

        mu_ = mu;
      }

      void set_sigma_tilde(const Eigen::VectorXd& sigma_tilde) {
        static const char* function =
          "stan::variational::advi_params_meanfield::set_sigma_tilde";

        stan::math::check_size_match(function,
                               "Dimension of input vector", sigma_tilde.size(),
                               "Dimension of current vector", dimension_ );

        sigma_tilde_ = sigma_tilde;
      }

      // Entropy of normal: 0.5 * log det diag(sigma^2) = sum(log(sigma))
      //                                                = sum(sigma_tilde)
      double entropy() const {
        return sigma_tilde_.sum();
      }

      // // Calculate natural parameters
      // Eigen::VectorXd nat_params() const {

      //   // Compute the variance
      //   Eigen::VectorXd variance = sigma_tilde_.array().exp().square();

      //   // Create a vector twice the dimension size
      //   Eigen::VectorXd natural_params(2*dimension_);

      //   // Concatenate the natural parameters
      //   natural_params << mu_.array().cwiseQuotient(variance.array()),
      //                     variance.array().cwiseInverse();

      //   return natural_params;
      // }

      // Implement f^{-1}(\check{z}) = sigma * \check{z} + \mu
      Eigen::VectorXd to_unconstrained(const Eigen::VectorXd& z_check) const {
        static const char* function = "stan::variational::advi_params_meanfield"
                                      "::to_unconstrained";

        stan::math::check_size_match(function,
                         "Dimension of mean vector", dimension_,
                         "Dimension of input vector", z_check.size() );

        // exp(sigma_tilde) * z_check + mu
        return z_check.array().cwiseProduct(sigma_tilde_.array().exp())
               + mu_.array();
      };

    };

  } // variational

} // stan

#endif
