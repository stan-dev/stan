#ifndef STAN__VB__VB_PARAMS_MEANFIELD__HPP
#define STAN__VB__VB_PARAMS_MEANFIELD__HPP

#include <vector>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/functions/max.hpp>

#include <stan/math/error_handling/matrix/check_size_match.hpp>
#include <stan/math/error_handling/check_not_nan.hpp>
#include <stan/math/error_handling/check_positive_finite.hpp>

namespace stan {

  namespace vb {

    class vb_params_meanfield {

    private:

      Eigen::VectorXd mu_;         // Mean of location-scale family
      Eigen::VectorXd sigma2_; // Log standard deviations
      int dimension_;

    public:

      vb_params_meanfield(Eigen::VectorXd const& mu,
                          Eigen::VectorXd const& sigma2) :
      mu_(mu),
      sigma2_(sigma2),
      dimension_(mu.size()) {

        static const char* function = "stan::vb::vb_params_meanfield(%1%)";

        double tmp(0.0);
        stan::math::check_size_match(function,
                                 dimension_,     "Dimension of mean vector",
                                 sigma2_.size(), "Dimension of variance vector",
                                 &tmp);

      };

      virtual ~vb_params_meanfield() {}; // No-op

      // Accessors
      int dimension() const { return dimension_; }
      Eigen::VectorXd const& mu()     const { return mu_; }
      Eigen::VectorXd const& sigma2() const { return sigma2_; }

      // Mutators
      void set_mu(Eigen::VectorXd const& mu)         { mu_ = mu; }
      void set_sigma2(Eigen::VectorXd const& sigma2) {
        sigma2_ = sigma2;
      }

      // Implements f^{-1}(\check{z}) = sigma * \check{z} + \mu
      Eigen::VectorXd to_unconstrained(Eigen::VectorXd const& z_check) const {
        static const char* function = "stan::vb::vb_params_meanfield"
                                      "::to_unconstrained(%1%)";

        double tmp(0.0);
        stan::math::check_size_match(function,
                         dimension_,     "Dimension of mean vector",
                         z_check.size(), "Dimension of input vector",
                         &tmp);


        // stan::math::check_positive_finite(function,
        //     sigma2_,  "sigma2_",
        //     &tmp);

        // return sigma2_.array().exp().cwiseProduct(z_check) + mu_;
        return z_check.array().cwiseProduct(sigma2_.array().exp()) + mu_.array();
      };

    };

  } // vb

} // stan

#endif
