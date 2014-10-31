#ifndef STAN__VB__BBVB__HPP
#define STAN__VB__BBVB__HPP

#include <ostream>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/log_determinant.hpp>
#include <stan/model/util.hpp>

#include <stan/math/functions.hpp>  // I had to add these two lines beceause
#include <stan/math/matrix.hpp>     // the unit tests wouldn't compile...

#include <stan/prob/distributions/multivariate/continuous/multi_normal.hpp>

#include <stan/math/error_handling/matrix/check_size_match.hpp>
#include <stan/math/error_handling/matrix/check_square.hpp>

#include <stan/math/error_handling/check_not_nan.hpp>

#include <stan/vb/base_vb.hpp>
#include <stan/vb/vb_params_fullrank.hpp>
#include <stan/vb/vb_params_meanfield.hpp>

namespace stan {

  namespace vb {

    template <class M, class BaseRNG>
    class bbvb : public base_vb {

    public:

      bbvb(M& m,
           Eigen::VectorXd& cont_params,
           double& elbo,
           BaseRNG& rng,
           std::ostream* o, std::ostream* e):
        base_vb(o, e, "bbvb"),
        model_(m),
        cont_params_(cont_params),
        elbo_(elbo),
        rng_(rng),
        n_monte_carlo_(5e1) {};

      virtual ~bbvb() {};


      /**
       * FULL-RANK ELBO
       *
       * Calculates the "blackbox" Evidence Lower BOund (ELBO) by sampling
       * from the standard multivariate normal (for now), affine transform
       * the sample, and evaluating the log joint, adjusted by the entropy
       * term of the normal, which is proportional to 0.5*logdet(L^T L)
       *
       * @param   muL   mean and cholesky factor of affine transform
       * @return        evidence lower bound (elbo)
       */
      double calc_ELBO(vb_params_fullrank const& muL) {
        double elbo(0.0);
        int dim = muL.dimension();

        Eigen::VectorXd z_check   = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_tilde   = Eigen::VectorXd::Zero(dim);
        Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1> z_tilde_var(dim);

        for (int i = 0; i < n_monte_carlo_; ++i) {
          // Draw from standard normal and transform to unconstrained space
          for (int d = 0; d < dim; ++d) {
            z_check(d) = stan::prob::normal_rng(0,1,rng_);
          }
          z_tilde = muL.to_unconstrained(z_check);

          // FIXME: is this the right way to do this?
          //
          // We need to call the stan::agrad::var version of log_prob
          // to get the correct proportionality and Jacobian terms
          for (int var_index = 0; var_index < dim; ++var_index) {
            z_tilde_var(var_index) = stan::agrad::var(z_tilde(var_index));
          }
          elbo += (model_.template
                   log_prob<true,true>(z_tilde_var, &std::cout)).val();
          // END of FIXME
        }
        elbo /= static_cast<double>(n_monte_carlo_);

        // Entropy of normal: 0.5 * log det (L^T L) = 0.5 * 2 * sum(log(diag(L)))
        elbo += stan::math::sum(stan::math::log(
                                stan::math::diagonal(muL.L_chol()))
                                               );

        return elbo;
      }


      /**
       * MEAN-FIELD ELBO
       *
       * Calculates the "blackbox" Evidence Lower BOund (ELBO) by sampling
       * from the standard multivariate normal (for now), affine transform
       * the sample, and evaluating the log joint, adjusted by the entropy
       * term of the normal, which is proportional to 0.5*logdet(L^T L)
       *
       * @param   musigma2   mean and variance vector of affine transform
       * @return             evidence lower bound (elbo)
       */
      double calc_ELBO(vb_params_meanfield const& musigma2) {
        double elbo(0.0);
        int dim = musigma2.dimension();

        Eigen::VectorXd z_check   = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_tilde   = Eigen::VectorXd::Zero(dim);
        Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1> z_tilde_var(dim);

        for (int i = 0; i < n_monte_carlo_; ++i) {
          // Draw from standard normal and transform to unconstrained space
          for (int d = 0; d < dim; ++d) {
            z_check(d) = stan::prob::normal_rng(0,1,rng_);
          }
          z_tilde = musigma2.to_unconstrained(z_check);

          // FIXME: is this the right way to do this?
          //
          // We need to call the stan::agrad::var version of log_prob
          // to get the correct proportionality and Jacobian terms
          for (int var_index = 0; var_index < dim; ++var_index) {
            z_tilde_var(var_index) = stan::agrad::var(z_tilde(var_index));
          }
          elbo += (model_.template
                   log_prob<true,true>(z_tilde_var, &std::cout)).val();
          // END of FIXME
        }
        elbo /= static_cast<double>(n_monte_carlo_);

        // Entropy of normal: 0.5 * log det diag(sigma^2) = 0.5 * sum(log(sigma^2))
        elbo += 0.5 * stan::math::sum( stan::math::log(musigma2.sigma2()) );

        return elbo;
      }


      /**
       * FULL-RANK GRADIENTS
       *
       * Calculates the "blackbox" gradient with respect to BOTH the location
       * vector (mu) and the cholesky factor of the scale matrix (L) in
       * parallel. It uses the same gradient computed from a set of Monte Carlo
       * samples
       *
       * @param muL     mean and cholesky factor of affine transform
       * @param mu_grad gradient of location vector parameter
       * @param L_grad  gradient of scale matrix parameter
       */
      void calc_combined_grad(
        vb_params_fullrank const& muL,
        Eigen::VectorXd& mu_grad,
        Eigen::MatrixXd& L_grad) {
        static const char* function = "stan::vb::bbvb.calc_combined_grad(%1%)";

        int dim       = muL.dimension();
        double tmp_lp = 0.0;

        double tmp(0.0);
        stan::math::check_size_match(function,
                              mu_grad.size(),  "Dimension of mu grad vector",
                              dim, "Dimension of mean vector in variational q",
                              &tmp);
        stan::math::check_square(function, L_grad, "Scale matrix", &tmp);
        stan::math::check_size_match(function,
                              L_grad.rows(), "Dimension of scale matrix",
                              dim, "Dimension of mean vector in variational q",
                              &tmp);

        // Initialize everything to zero
        mu_grad = Eigen::VectorXd::Zero(dim);
        L_grad  = Eigen::MatrixXd::Zero(dim,dim);
        Eigen::VectorXd tmp_mu_grad = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_check = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_tilde = Eigen::VectorXd::Zero(dim);

        // Naive Monte Carlo integration
        for (int i = 0; i < n_monte_carlo_; ++i) {

          // Draw from standard normal and transform to unconstrained space
          for (int d = 0; d < dim; ++d) {
            z_check(d) = stan::prob::normal_rng(0,1,rng_);
          }
          z_tilde = muL.to_unconstrained(z_check);

          // Compute gradient step in unconstrained space
          stan::model::gradient(model_, z_tilde, tmp_lp, tmp_mu_grad,
                                &std::cout);

          // Update mu
          mu_grad += tmp_mu_grad;

          // Update L (lower triangular)
          for (int ii = 0; ii < dim; ++ii) {
            for (int jj = 0; jj <= ii; ++jj) {
              L_grad(ii,jj) += tmp_mu_grad(ii) * z_check(jj);
            }
          }

        }
        mu_grad /= static_cast<double>(n_monte_carlo_);
        L_grad  /= static_cast<double>(n_monte_carlo_);

        // Add gradient of entropy term
        L_grad.diagonal().array() += muL.L_chol().diagonal().array().inverse();
      }


      /**
       * MEAN-FIELD GRADIENTS
       *
       * Calculates the "blackbox" gradient with respect to BOTH the location
       * vector (mu) and the variance vector (sigma^2) in parallel.
       * It uses the same gradient computed from a set of Monte Carlo
       * samples
       *
       * @param musigma2     mean and cholesky factor of affine transform
       * @param mu_grad      gradient of location vector parameter
       * @param sigma2_grad  gradient of variance vector parameter
       */
      void calc_combined_grad(
        vb_params_meanfield const& musigma2,
        Eigen::VectorXd& mu_grad,
        Eigen::VectorXd& sigma2_grad) {
        static const char* function = "stan::vb::bbvb.calc_combined_grad(%1%)";

        int dim       = musigma2.dimension();
        double tmp_lp = 0.0;

        double tmp(0.0);
        stan::math::check_size_match(function,
                              mu_grad.size(),  "Dimension of mu grad vector",
                              dim, "Dimension of mean vector in variational q",
                              &tmp);

        // Initialize everything to zero
        mu_grad      = Eigen::VectorXd::Zero(dim);
        sigma2_grad  = Eigen::VectorXd::Zero(dim);

        Eigen::VectorXd tmp_mu_grad = Eigen::VectorXd::Zero(dim);

        Eigen::VectorXd z_check = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_tilde = Eigen::VectorXd::Zero(dim);

        // Naive Monte Carlo integration
        for (int i = 0; i < n_monte_carlo_; ++i) {

          // Draw from standard normal and transform to unconstrained space
          for (int d = 0; d < dim; ++d) {
            z_check(d) = stan::prob::normal_rng(0,1,rng_);
          }
          z_tilde = musigma2.to_unconstrained(z_check);

          stan::math::check_not_nan(function,
            z_tilde,  "z_tilde",
            &tmp);

          // Compute gradient step in unconstrained space
          stan::model::gradient(model_, z_tilde, tmp_lp, tmp_mu_grad,
                                &std::cout);

          // Update mu
          mu_grad.array() = mu_grad.array() + tmp_mu_grad.array();

          // Update sigma2
          sigma2_grad.array() = sigma2_grad.array() + tmp_mu_grad.dot(z_check);

        }
        mu_grad      /= static_cast<double>(n_monte_carlo_);
        sigma2_grad  /= static_cast<double>(n_monte_carlo_);

        // Add gradient of entropy term
        sigma2_grad.array() += (
                                musigma2.sigma2().array()
                                ).inverse();
      }


      /**
       * FULL-RANK ROBBINS-MONRO
       *
       * Runs Robbins-Monro Stochastic Gradient for some number of iterations
       *
       * @param muL            mean and cholesky factor of affine transform
       * @param max_iterations number of iterations to run algorithm
       */
      void do_robbins_monro_adagrad( vb_params_fullrank& muL,
                                     int max_iterations ) {
        Eigen::VectorXd mu_grad = Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::MatrixXd L_grad  = Eigen::MatrixXd::Zero(model_.num_params_r(),
                                                        model_.num_params_r());

        // ADAgrad parameters
        double eta = 1.0;
        double tau = 1.0;
        Eigen::VectorXd mu_s = Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::MatrixXd L_s  = Eigen::MatrixXd::Zero(model_.num_params_r(),
                                                     model_.num_params_r());

        // // rmsprop parameters
        // double window_size = 100.0;

        for (int i = 0; i < max_iterations; ++i)
        {
          if (out_stream_) *out_stream_ << "----------------" << std::endl
                                        << "  iter " << i << std::endl
                                        << "----------------" << std::endl;

          // Compute gradient using Monte Carlo integration
          calc_combined_grad(muL, mu_grad, L_grad);

          // Accumulate S vector for ADAgrad
          mu_s.array() += mu_grad.array().square();
          L_s.array()  += L_grad.array().square();

          // // Moving average for rmsprop
          // mu_s.array() = ( 1 - 1.0/window_size ) * mu_s.array()
          //                 + 1.0/window_size * mu_grad.array().square();
          // L_s.array()  = ( 1 - 1.0/window_size ) * L_s.array()
          //                 + 1.0/window_size * L_grad.array().square();


          // Take ADAgrad or rmsprop step
          muL.set_mu( muL.mu().array() +
            eta * mu_grad.array() / (tau + mu_s.array().sqrt()) );
          muL.set_L_chol(  muL.L_chol().array()  +
            eta * L_grad.array()  / (tau + L_s.array().sqrt()) );

          cont_params_ = muL.mu();
          if (out_stream_) *out_stream_ << "mu = " << std::endl
                                        << muL.mu() << std::endl;

          // if (out_stream_) *out_stream_ << "L_chol = " << std::endl
          //                               << muL.L_chol() << std::endl;

          // if (out_stream_) *out_stream_ << "Sigma = " << std::endl
          //                               << muL.L_chol() * muL.L_chol().transpose() << std::endl;

          // elbo_ = calc_ELBO(muL);
          // if (out_stream_) *out_stream_ << "elbo_ = " << elbo_ << std::endl;

        }
      }


      /**
       * MEAN-FIELD ROBBINS-MONRO
       *
       * Runs Robbins-Monro Stochastic Gradient for some number of iterations
       *
       * @param musigma2        mean and variance vector of affine transform
       * @param max_iterations  number of iterations to run algorithm
       */
      void do_robbins_monro_adagrad( vb_params_meanfield& musigma2,
                                     int max_iterations ) {
        Eigen::VectorXd mu_grad      = Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::VectorXd sigma2_grad  = Eigen::VectorXd::Zero(model_.num_params_r());

        // ADAgrad parameters
        double eta = 1.0;
        double tau = 1.0;
        Eigen::VectorXd mu_s     = Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::VectorXd sigma2_s = Eigen::VectorXd::Zero(model_.num_params_r());


        for (int i = 0; i < max_iterations; ++i)
        {
          if (out_stream_) *out_stream_ << "----------------" << std::endl
                                        << "  iter " << i << std::endl
                                        << "----------------" << std::endl;

          // Compute gradient using Monte Carlo integration
          calc_combined_grad(musigma2, mu_grad, sigma2_grad);

          // Accumulate S vector for ADAgrad
          mu_s.array()      += mu_grad.array().square();
          sigma2_s.array()  += sigma2_grad.array().square();



          // Take ADAgrad or rmsprop step
          musigma2.set_mu( musigma2.mu().array() +
            eta * mu_grad.array() / (tau + mu_s.array().sqrt()) );
          musigma2.set_sigma2(  musigma2.sigma2().array()  +
            eta * sigma2_grad.array()  / (tau + sigma2_s.array().sqrt()) );

          cont_params_ = musigma2.mu();
          if (out_stream_) *out_stream_ << "mu = " << std::endl
                                        << musigma2.mu() << std::endl;

          // if (out_stream_) *out_stream_ << "sigma2 = " << std::endl
          //                               << musigma2.sigma2() << std::endl;

          // elbo_ = calc_ELBO(muL);
          // if (out_stream_) *out_stream_ << "elbo_ = " << elbo_ << std::endl;

        }
      }

      void run_robbins_monro_fullrank() {
        if (out_stream_) *out_stream_
          << "This is base_vb::bbvb::run_robbins_monro_fullrank()" << std::endl;
        if (out_stream_) *out_stream_
          << "cont_params_ = " << std::endl
          << cont_params_ << std::endl << std::endl;

        // Initialize variational parameters: mu, L
        Eigen::VectorXd mu = cont_params_;
        Eigen::MatrixXd L  = Eigen::MatrixXd::Identity(model_.num_params_r(),
                                                       model_.num_params_r());
        vb_params_fullrank muL = vb_params_fullrank(mu,L);

        // Robbins Monro ADAgrad
        do_robbins_monro_adagrad(muL, 1000);

        return;
      }

      void run_robbins_monro_meanfield() {
        if (out_stream_) *out_stream_
          << "This is base_vb::bbvb::run_robbins_monro_meanfield()" << std::endl;
        if (out_stream_) *out_stream_
          << "cont_params_ = " << std::endl
          << cont_params_ << std::endl << std::endl;

        // Initialize variational parameters: mu, L
        Eigen::VectorXd mu = cont_params_;
        Eigen::MatrixXd sigma2  = Eigen::VectorXd::Ones(model_.num_params_r());

        vb_params_meanfield mu_sigma2 = vb_params_meanfield(mu,sigma2);

        // Robbins Monro ADAgrad
        do_robbins_monro_adagrad(mu_sigma2, 1500);

        return;
      }

    protected:

      M& model_;
      Eigen::VectorXd& cont_params_;
      double elbo_;
      BaseRNG& rng_;
      int n_monte_carlo_;

    };

  } // vb

} // stan

#endif

