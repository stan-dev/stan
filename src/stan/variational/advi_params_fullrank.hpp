#ifndef STAN__VB__VB_PARAMS_FULLRANK__HPP
#define STAN__VB__VB_PARAMS_FULLRANK__HPP

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/err/check_cholesky_factor.hpp>
#include <stan/math/prim/mat/fun/LDLT_factor.hpp>

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

        stan::math::check_square(function, "Cholesky factor", L_chol_);
        stan::math::check_size_match(function,
                               "Dimension of mean vector",     dimension_,
                               "Dimension of Cholesky factor", L_chol_.rows() );
        stan::math::check_cholesky_factor(function,
                                 "Cholesky factor", L_chol_);

      };

      virtual ~vb_params_fullrank() {}; // No-op

      // Accessors
      int dimension() const { return dimension_; }
      Eigen::VectorXd const& mu()     const { return mu_; }
      Eigen::MatrixXd const& L_chol() const { return L_chol_; }

      // Mutators
      void set_mu(Eigen::VectorXd const& mu) { mu_ = mu; }
      void set_L_chol(Eigen::MatrixXd const& L_chol) { L_chol_ = L_chol; }

      // Entropy of normal: 0.5 * log det (L^T L) = sum(log(abs(diag(L))))
      double entropy() const {
        double tmp(0.0);
        double result(0.0);
        for (int d = 0; d < dimension_; ++d) {
          tmp = abs(L_chol_(d,d));
          if (tmp != 0.0) {
            result += log(tmp);
          }
        }
        return result;
      }

      // Calculate natural parameters
      Eigen::VectorXd nat_params() const {

        // FIXME: stupid Eigen. can't initialize LLT factor with L.
        // Compute the covariance matrix
        Eigen::MatrixXd Sigma = L_chol_ * L_chol_.transpose();
        stan::math::LDLT_factor<double,-1,-1> Sigma_LDLT(Sigma);

        // Create a vector twice the dimension size
        Eigen::VectorXd natural_params(dimension_ + dimension_^2);

        // Concatenate the natural parameters
        natural_params << (Sigma_LDLT.solve(mu_)).array(),
                          (Sigma_LDLT.solve(
                            Eigen::MatrixXd::Identity(dimension_,dimension_))
                          ).array();

        return natural_params;
      }

      // Implements f^{-1}(\check{z}) = L\check{z} + \mu
      Eigen::VectorXd to_unconstrained(Eigen::VectorXd const& z_check) const {
        static const char* function = "stan::vb::vb_params_fullrank"
                                      "::to_unconstrained(%1%)";

        stan::math::check_size_match(function,
                         "Dimension of input vector", z_check.size(),
                         "Dimension of mean vector",  dimension_ );

        return L_chol_*z_check + mu_;
      };

    };

  } // vb

} // stan

#endif
