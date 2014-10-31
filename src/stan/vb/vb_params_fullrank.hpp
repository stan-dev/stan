#ifndef STAN__VB__VB_PARAMS_FULLRANK__HPP
#define STAN__VB__VB_PARAMS_FULLRANK__HPP

#include <vector>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/matrix/check_size_match.hpp>
#include <stan/math/error_handling/matrix/check_square.hpp>
#include <stan/math/error_handling/matrix/check_cholesky_factor.hpp>

namespace stan {

  namespace vb {

    class vb_params_fullrank {

    private:

      Eigen::VectorXd mu_;     // Mean of location-scale family
      Eigen::MatrixXd L_chol_; // Cholesky factor of scale matrix
                               // NOTE: \Sigma = L_chol_ * L_chol_.transpose()
      int dimension_;

    public:

      vb_params_fullrank(Eigen::VectorXd const& mu,
                          Eigen::MatrixXd const& L_chol) :
      mu_(mu), L_chol_(L_chol), dimension_(mu.size()) {

        static const char* function = "stan::vb::vb_params_fullrank(%1%)";

        double tmp(0.0);
        stan::math::check_square(function, L_chol_, "Cholesky factor", &tmp);
        stan::math::check_size_match(function,
                                 dimension_,     "Dimension of mean vector",
                                 L_chol_.rows(), "Dimension of Cholesky factor",
                                 &tmp);
        stan::math::check_cholesky_factor(function,
                                 L_chol_, "Cholesky factor", &tmp);

      };

      virtual ~vb_params_fullrank() {}; // No-op

      // Accessors
      int dimension() const { return dimension_; }
      Eigen::VectorXd const& mu()     const { return mu_; }
      Eigen::MatrixXd const& L_chol() const { return L_chol_; }

      // Mutators
      void set_mu(Eigen::VectorXd const& mu) { mu_ = mu; }
      void set_L_chol(Eigen::MatrixXd const& L_chol) { L_chol_ = L_chol; }

      // Implements f^{-1}(\check{z}) = L\check{z} + \mu
      Eigen::VectorXd to_unconstrained(Eigen::VectorXd const& z_check) const {
        static const char* function = "stan::vb::vb_params_fullrank"
                                      "::to_unconstrained(%1%)";

        double tmp(0.0);
        stan::math::check_size_match(function,
                         z_check.size(), "Dimension of input vector",
                         dimension_,     "Dimension of mean vector",
                         &tmp);

        return L_chol_*z_check + mu_;
      };

    };

  } // vb

} // stan

#endif
