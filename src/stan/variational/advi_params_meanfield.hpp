#ifndef STAN_VARIATIONAL_ADVI_PARAMS_MEANFIELD__HPP
#define STAN_VARIATIONAL_ADVI_PARAMS_MEANFIELD__HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>

#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <vector>

namespace stan {

  namespace variational {

    class advi_params_meanfield {
    private:
      Eigen::VectorXd mu_;     // Mean vector
      Eigen::VectorXd omega_;  // Log standard deviation vector
      int dimension_;

    public:
      advi_params_meanfield(const Eigen::VectorXd& mu,
                            const Eigen::VectorXd& omega) :
      mu_(mu), omega_(omega), dimension_(mu.size()) {
        static const char* function =
          "stan::variational::advi_params_meanfield";

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
          "stan::variational::advi_params_meanfield::set_mu";

        stan::math::check_size_match(function,
                               "Dimension of input vector", mu.size(),
                               "Dimension of current vector", dimension_);
        stan::math::check_not_nan(function, "Input vector", mu);
        mu_ = mu;
      }

      void set_omega(const Eigen::VectorXd& omega) {
        static const char* function =
          "stan::variational::advi_params_meanfield::set_omega";

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
      Eigen::VectorXd loc_scale_transform(const Eigen::VectorXd& eta) const {
        static const char* function = "stan::variational::advi_params_meanfield"
                                      "::loc_scale_transform";

        stan::math::check_size_match(function,
                         "Dimension of mean vector", dimension_,
                         "Dimension of input vector", eta.size() );
        stan::math::check_not_nan(function, "Input vector", eta);

        // exp(omega) * eta + mu
        return eta.array().cwiseProduct(omega_.array().exp()) + mu_.array();
      }
    };
  }  // variational
}  // stan

#endif
