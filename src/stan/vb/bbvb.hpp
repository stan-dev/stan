#ifndef STAN__VB__BBVB__HPP
#define STAN__VB__BBVB__HPP

#include <stan/math/matrix/Eigen.hpp>

#include <ostream>
#include <stan/common/write_iteration_csv.hpp>

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
        n_monte_carlo_(10) {};

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
        // static const char* function = "stan::vb::bbvb.calc_ELBO(%1%)";
        // double error_tmp(0.0);
        double elbo(0.0);
        int dim = muL.dimension();

        int elbo_n_monte_carlo(5);

        Eigen::VectorXd z_check   = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_tilde   = Eigen::VectorXd::Zero(dim);
        Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1> z_tilde_var(dim);

        for (int i = 0; i < elbo_n_monte_carlo; ++i) {
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
        elbo /= static_cast<double>(elbo_n_monte_carlo);

        // Entropy of normal: 0.5 * log det (L^T L) = sum(log(abs(diag(L))))
        double tmp(0.0);
        for (int d = 0; d < dim; ++d) {
          tmp = abs(muL.L_chol()(d,d));
          if (tmp != 0.0) {
            elbo += log(tmp);
          }
        }

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
       * @param   musigmatilde   mean and log-std vector of affine transform
       * @return                 evidence lower bound (elbo)
       */
      double calc_ELBO(vb_params_meanfield const& musigmatilde) {
        double elbo(0.0);
        int dim = musigmatilde.dimension();

        int elbo_n_monte_carlo(5);

        Eigen::VectorXd z_check   = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_tilde   = Eigen::VectorXd::Zero(dim);
        Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1> z_tilde_var(dim);

        for (int i = 0; i < elbo_n_monte_carlo; ++i) {
          // Draw from standard normal and transform to unconstrained space
          for (int d = 0; d < dim; ++d) {
            z_check(d) = stan::prob::normal_rng(0,1,rng_);
          }
          z_tilde = musigmatilde.to_unconstrained(z_check);

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
        elbo /= static_cast<double>(elbo_n_monte_carlo);

        // Entropy of normal: 0.5 * log det diag(sigma^2) = sum(log(sigma))
        //                                                = sum(sigma_tilde)
        elbo += stan::math::sum( musigmatilde.sigma_tilde() );

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
       * @param musigmatilde      mean and log-std vector of affine transform
       * @param mu_grad           gradient of mean vector parameter
       * @param sigma_tilde_grad  gradient of log-std vector parameter
       */
      void calc_combined_grad(
        vb_params_meanfield const& musigmatilde,
        Eigen::VectorXd& mu_grad,
        Eigen::VectorXd& sigma_tilde_grad) {
        static const char* function = "stan::vb::bbvb.calc_combined_grad(%1%)";

        int dim       = musigmatilde.dimension();
        double tmp_lp = 0.0;

        double tmp(0.0);
        stan::math::check_size_match(function,
                              mu_grad.size(),  "Dimension of mu grad vector",
                              dim, "Dimension of mean vector in variational q",
                              &tmp);

        // Initialize everything to zero
        mu_grad          = Eigen::VectorXd::Zero(dim);
        sigma_tilde_grad = Eigen::VectorXd::Zero(dim);

        Eigen::VectorXd tmp_mu_grad = Eigen::VectorXd::Zero(dim);

        Eigen::VectorXd z_check = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_tilde = Eigen::VectorXd::Zero(dim);

        // Naive Monte Carlo integration
        for (int i = 0; i < n_monte_carlo_; ++i) {

          // Draw from standard normal and transform to unconstrained space
          for (int d = 0; d < dim; ++d) {
            z_check(d) = stan::prob::normal_rng(0,1,rng_);
          }
          z_tilde = musigmatilde.to_unconstrained(z_check);

          stan::math::check_not_nan(function,
            z_tilde,  "z_tilde",
            &tmp);

          // Compute gradient step in unconstrained space
          stan::model::gradient(model_, z_tilde, tmp_lp, tmp_mu_grad,
                                &std::cout);

          // Update mu
          mu_grad.array() = mu_grad.array() + tmp_mu_grad.array();

          // Update sigma_tilde
          sigma_tilde_grad.array() = sigma_tilde_grad.array()
            + tmp_mu_grad.array().cwiseProduct(z_check.array());

        }
        mu_grad           /= static_cast<double>(n_monte_carlo_);
        sigma_tilde_grad  /= static_cast<double>(n_monte_carlo_);

        // multiply by exp(sigma_tilde)
        sigma_tilde_grad.array() =
          sigma_tilde_grad.array().cwiseProduct(
                                      musigmatilde.sigma_tilde().array().exp());

        // Add gradient of entropy term (just equal to element-wise 1 here)
        sigma_tilde_grad.array() += 1.0;
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
        double eta = 0.1;
        double tau = 1.0;
        Eigen::VectorXd mu_s = Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::MatrixXd L_s  = Eigen::MatrixXd::Zero(model_.num_params_r(),
                                                     model_.num_params_r());

        // rmsprop parameters
        double window_size = 100.0;

        std::vector<double> print_vector;

        for (int i = 0; i < max_iterations; ++i)
        {
          print_vector.clear();

          std::cout
          // << "----------------" << std::endl
          << "  iter " << i     << std::endl;
          // << "----------------" << std::endl;

          // Compute gradient using Monte Carlo integration
          calc_combined_grad(muL, mu_grad, L_grad);

          // Accumulate S vector for ADAgrad
          mu_s.array() += mu_grad.array().square();
          L_s.array()  += L_grad.array().square();

          // Moving average for rmsprop
          mu_s.array() = ( 1 - 1.0/window_size ) * mu_s.array()
                          + 1.0/window_size * mu_grad.array().square();
          L_s.array()  = ( 1 - 1.0/window_size ) * L_s.array()
                          + 1.0/window_size * L_grad.array().square();


          // Take ADAgrad or rmsprop step
          muL.set_mu( muL.mu().array() +
            eta * mu_grad.array() / (tau + mu_s.array().sqrt()) );
          muL.set_L_chol(  muL.L_chol().array()  +
            eta * L_grad.array()  / (tau + L_s.array().sqrt()) );

          // print out to std::cout for now
          // std::cout
          // << "mu = "  << std::endl
          // << muL.mu() << std::endl;

          // std::cout
          // << "L_chol = "  << std::endl
          // << muL.L_chol() << std::endl;

          // write elbo and parameters to "error stream"
          // if (err_stream_){
          //   for (int d = 0; d < muL.dimension(); ++d) {
          //     print_vector.push_back(muL.mu()(d));
          //   }
          //   stan::common::write_iteration_csv(
          //     *err_stream_, calc_ELBO(muL), print_vector);
          // }
          if (err_stream_) {
            if ((i < 10 || (i<100 && i%10==0) || i%100==0)){
              print_vector.push_back(calc_ELBO(muL));
              stan::common::write_iteration_csv(*err_stream_, i , print_vector);
            }
          }


          // std::cout << "Sigma = " << std::endl
          //                               << muL.L_chol() * muL.L_chol().transpose() << std::endl;


        }
      }


      /**
       * MEAN-FIELD ROBBINS-MONRO
       *
       * Runs Robbins-Monro Stochastic Gradient for some number of iterations
       *
       * @param musigmatilde    mean and log-std vector of affine transform
       * @param max_iterations  number of iterations to run algorithm
       */
      void do_robbins_monro_adagrad( vb_params_meanfield& musigmatilde,
                                     int max_iterations ) {

        Eigen::VectorXd mu_grad           = Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::VectorXd sigma_tilde_grad  = Eigen::VectorXd::Zero(model_.num_params_r());

        // ADAgrad parameters
        double eta = 1.0;
        double tau = 1.0;
        Eigen::VectorXd mu_s          = Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::VectorXd sigma_tilde_s = Eigen::VectorXd::Zero(model_.num_params_r());

        // RMSprop window_size
        double window_size = 100.0;

        std::vector<double> print_vector;

        for (int i = 0; i < max_iterations; ++i)
        {
          print_vector.clear();

          std::cout
          // << "----------------" << std::endl
          << "  iter " << i     << std::endl;
          // << "----------------" << std::endl;

          // Compute gradient using Monte Carlo integration
          calc_combined_grad(musigmatilde, mu_grad, sigma_tilde_grad);

          // Accumulate S vector for ADAgrad
          mu_s.array()           += mu_grad.array().square();
          sigma_tilde_s.array()  += sigma_tilde_grad.array().square();

          mu_s.array() = ( 1.0 - 1.0/window_size ) * mu_s.array()
                          + 1.0/window_size * mu_grad.array().square();
          sigma_tilde_s.array()  = ( 1.0 - 1.0/window_size ) * sigma_tilde_s.array()
                          + 1.0/window_size * sigma_tilde_grad.array().square();

          // Take ADAgrad or rmsprop step
          musigmatilde.set_mu(
            musigmatilde.mu().array() +
            eta * mu_grad.array() / (tau + mu_s.array().sqrt())
            );
          musigmatilde.set_sigma_tilde(
            musigmatilde.sigma_tilde().array()  +
            eta * sigma_tilde_grad.array()  / (tau + sigma_tilde_s.array().sqrt())
            );

          // print out to std::cout for now
          // std::cout
          // << "mu = " << std::endl
          // << musigmatilde.mu() << std::endl;

          // write elbo and parameters to "error stream"
          // if ((i < 100 || i % 100 == 0) && err_stream_){
          if (err_stream_) {
            if ((i < 10 || (i<100 && i%10==0) || i%100==0)){
              print_vector.push_back(calc_ELBO(musigmatilde));
              stan::common::write_iteration_csv(*err_stream_, i , print_vector);
            }
          }


          // std::cout << "sigma_tilde = " << std::endl
          //                               << musigmatilde.sigma_tilde() << std::endl;

        }
      }

      void run_robbins_monro_fullrank() {
        std::cout
        << "This is base_vb::bbvb::run_robbins_monro_fullrank()" << std::endl;
        std::cout
        << "cont_params_ = " << std::endl
        << cont_params_ << std::endl << std::endl;

        // Initialize variational parameters: mu, L
        Eigen::VectorXd mu = cont_params_;
        Eigen::MatrixXd L  = Eigen::MatrixXd::Identity(model_.num_params_r(),
                                                       model_.num_params_r());
        vb_params_fullrank muL = vb_params_fullrank(mu,L);

        // Robbins Monro ADAgrad
        do_robbins_monro_adagrad(muL, 1000);

        cont_params_ = muL.mu();

        std::cout
        << "mu = " << std::endl
        << muL.mu() << std::endl;

        std::cout
        << "Sigma = " << std::endl
        << muL.L_chol() * muL.L_chol().transpose() << std::endl;

        return;
      }

      void run_robbins_monro_meanfield() {
        std::cout
        << "This is base_vb::bbvb::run_robbins_monro_meanfield()" << std::endl;
        std::cout
        << "cont_params_ = " << std::endl
        << cont_params_ << std::endl << std::endl;

        // Initialize variational parameters: mu, sigma_tilde
        Eigen::VectorXd mu           = cont_params_;
        Eigen::MatrixXd sigma_tilde  = Eigen::VectorXd::Constant(
                                                model_.num_params_r(),
                                                1.0
                                                );

        vb_params_meanfield musigmatilde = vb_params_meanfield(mu,sigma_tilde);

        // Robbins Monro ADAgrad
        do_robbins_monro_adagrad(musigmatilde, 1000);

        cont_params_ = musigmatilde.mu();

        std::cout
        << "mu = " << std::endl
        << musigmatilde.mu() << std::endl;

        std::cout
        << "sigma_tilde = " << std::endl
        << musigmatilde.sigma_tilde() << std::endl;

        return;
      }

      Eigen::VectorXd const& cont_params() {
        return cont_params_;
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

