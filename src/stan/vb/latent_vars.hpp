#ifndef STAN__VB__LATENT_VARS__HPP
#define STAN__VB__LATENT_VARS__HPP

#include <vector>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/mdivide_left_tri_low.hpp>
#include <stan/math/error_handling/matrix/check_size_match.hpp>
#include <stan/math/error_handling/matrix/check_square.hpp>
#include <stan/math/error_handling/matrix/check_cholesky_factor.hpp>

namespace stan {

  namespace vb {

    class latent_vars {

    private:

      Eigen::VectorXd mu_;     // Mean of location-scale family
      Eigen::MatrixXd L_chol_; // Cholesky factor of scale matrix
                               // NOTE: \Sigma = L_chol_ * L_chol_.transpose()
      // Eigen::MatrixXd L_log_;  // Log-Cholesky factor of scale matrix
      int dimension_;

    public:

      latent_vars(Eigen::VectorXd const& mu, Eigen::MatrixXd const& L_chol) :
      mu_(mu), L_chol_(L_chol), dimension_(mu.size()) {

        static const char* function = "stan::vb::latent_vars(%1%)";

        double tmp(0.0);
        stan::math::check_square(function, L_chol_, "Cholesky factor", &tmp);
        stan::math::check_size_match(function,
                                     L_chol_.rows(), "Dimension of Cholesky factor",
                                     dimension_,     "Dimension of mean vector",
                                     &tmp);
        stan::math::check_cholesky_factor(function, L_chol_, "Cholesky factor", &tmp);

      };

      virtual ~latent_vars() {}; // No-op

      // Accessors
      int dimension() const { return dimension_; }
      Eigen::VectorXd const& mu()     const { return mu_; }
      Eigen::MatrixXd const& L_chol() const { return L_chol_; }
      // Eigen::MatrixXd const& L_log()  const { return L_log_; }

      // Mutators
      void set_mu(Eigen::VectorXd const& mu) { mu_ = mu; }

      void set_L_chol(Eigen::MatrixXd const& L_chol) {
        L_chol_ = L_chol;
        // update_L_log();
      }

      // void set_L_log(Eigen::MatrixXd const& L_log) {
      //   L_log_ = L_log;
      //   update_L_chol();
      // }

      // // Updates the Cholesky factor from the log-Cholesky factor
      // void update_L_chol() {
      //   L_chol_ = L_log_;
      //   L_chol_.diagonal() = (L_log_.diagonal().array().exp() - 1.0).log();
      // }

      // // Updates the log-Cholesky factor from the Cholesky factor
      // void update_L_log() {
      //   L_log_ = L_chol_;
      //   L_log_.diagonal() = (L_chol_.diagonal().array().exp() + 1.0).log();
      // }

      // Implements f^{-1}(\check{z}) = L\check{z} + \mu
      Eigen::VectorXd to_unconstrained(Eigen::VectorXd const& z_check) const {
        static const char* function = "stan::vb::latent_vars"
                                      "::to_unconstrained(%1%)";

        double tmp(0.0);
        stan::math::check_size_match(function,
                         z_check.size(), "Dimension of input vector",
                         dimension_, "Dimension of mean vector",
                         &tmp);

        return L_chol_*z_check + mu_;
      };

      // // Implements g(\widetilde{z}) = L^{-1}(\check{z} - \mu)
      // void to_standardized(Eigen::VectorXd& x) const {
      //   static const char* function = "stan::vb::latent_vars"
      //                                 "::to_standardized(%1%)";

      //   double tmp(0.0);
      //   stan::math::check_size_match(function,
      //                    x.size(), "Dimension of input vector",
      //                    dimension_, "Dimension of mean vector",
      //                    &tmp);

      //   Eigen::VectorXd x_minus_mu = x - mu_;
      //   x = stan::math::mdivide_left_tri_low(L_, x_minus_mu);
      // };

    };

  } // vb

} // stan

#endif
