#ifndef STAN__VB__VB_PARAMS_MEANFIELD__HPP
#define STAN__VB__VB_PARAMS_MEANFIELD__HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/functions/max.hpp>
#include <stan/error_handling/matrix/check_size_match.hpp>

namespace stan {

  namespace vb {

    class vb_params_meanfield {

    private:

      Eigen::VectorXd mu_;          // Mean vector
      Eigen::VectorXd sigma_tilde_; // Log standard deviation vector
      int dimension_;

    public:

      vb_params_meanfield(Eigen::VectorXd const& mu,
                          Eigen::VectorXd const& sigma_tilde) :
      mu_(mu),
      sigma_tilde_(sigma_tilde),
      dimension_(mu.size()) {

        static const char* function = "stan::vb::vb_params_meanfield(%1%)";

        stan::error_handling::check_size_match(function,
                               "Dimension of mean vector", dimension_,
                               "Dimension of std vector", sigma_tilde_.size() );

      };

      virtual ~vb_params_meanfield() {}; // No-op

      // Accessors
      int dimension() const { return dimension_; }
      Eigen::VectorXd const& mu()          const { return mu_; }
      Eigen::VectorXd const& sigma_tilde() const { return sigma_tilde_; }

      // Mutators
      void set_mu(Eigen::VectorXd const& mu) { mu_ = mu; }
      void set_sigma_tilde(Eigen::VectorXd const& sigma_tilde) {
        sigma_tilde_ = sigma_tilde;
      }

      // Implements f^{-1}(\check{z}) = sigma * \check{z} + \mu
      Eigen::VectorXd to_unconstrained(Eigen::VectorXd const& z_check) const {
        static const char* function = "stan::vb::vb_params_meanfield"
                                      "::to_unconstrained(%1%)";

        stan::error_handling::check_size_match(function,
                         "Dimension of mean vector", dimension_,
                         "Dimension of input vector", z_check.size() );

        // exp(sigma_tilde) * z_check + mu
        return z_check.array().cwiseProduct(sigma_tilde_.array().exp())
               + mu_.array();
      };

    };

  } // vb

} // stan

#endif
