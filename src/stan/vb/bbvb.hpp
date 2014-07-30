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

#include <stan/vb/base_vb.hpp>
#include <stan/vb/latent_vars.hpp>

namespace stan {

  namespace vb {

    template <class M, class BaseRNG>
    class bbvb : public base_vb {

    public:

      bbvb(M& m,
           Eigen::VectorXd const& cont_params,
           BaseRNG& rng,
           std::ostream* o, std::ostream* e):
        base_vb(o, e, "bbvb"),
        model_(m),
        cont_params_(cont_params),
        rng_(rng),
        n_monte_carlo_(1e4) {};

      virtual ~bbvb() {};

      /**
       * Calculates the "blackbox" Evidence Lower BOund (ELBO) by sampling
       * from the standard multivariate normal (for now), affine transform
       * the sample, and evaluating the log joint, adjusted by the entrop
       * term of the normal, which is proportional to 0.5*logdet(L^T L)
       *
       * @param   muL   mean and cholesky factor of affine transform
       * @return        evidence lower bound (elbo)
       */
      double calc_ELBO(latent_vars const& muL) {
        double elbo(0.0);
        int dim = muL.dimension();

        Eigen::VectorXd zero_mean = Eigen::VectorXd::Zero(dim);
        Eigen::MatrixXd eye       = Eigen::MatrixXd::Identity(dim, dim);

        Eigen::VectorXd z_check;
        Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1> z_check_var(dim);
        for (int i = 0; i < n_monte_carlo_; ++i) {
          // Draw from standard normal and transform to unconstrained space
          z_check = stan::prob::multi_normal_rng(zero_mean, eye, rng_);
          muL.to_unconstrained(z_check);

          // FIXME: is this the right way to do this?
          //
          // We need to call the stan::agrad::var version of log_prob
          // to get the correct proportionality and Jacobian terms
          for (int var_index = 0; var_index < dim; ++var_index)
          {
            z_check_var(var_index) = stan::agrad::var(z_check(var_index));
          }
          elbo += (model_.template log_prob<true,true>(z_check_var, &std::cout)).val();
          // END of FIXME
        }
        elbo /= static_cast<double>(n_monte_carlo_);

        // Entropy of normal: 0.5 * log det (L^T L) = 0.5 * 2 * sum(diag(L))
        elbo += stan::math::sum(stan::math::diagonal(muL.L()));

        return elbo;
      }


      void calc_mu_grad(latent_vars const& muL, Eigen::VectorXd& mu_grad) {
        static const char* function = "stan::vb::bbvb.calc_mu_grad(%1%)";

        int dim = muL.dimension();
        double tmp_lp = 0.0;

        double tmp(0.0);
        stan::math::check_size_match(function,
                              mu_grad.size(),  "Dimension of mu grad vector",
                              dim, "Dimension of mean vector in variational q",
                              &tmp);

        mu_grad                     = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd tmp_mu_grad = Eigen::VectorXd::Zero(dim);

        Eigen::VectorXd zero_mean = Eigen::VectorXd::Zero(dim);
        Eigen::MatrixXd eye       = Eigen::MatrixXd::Identity(dim, dim);

        Eigen::VectorXd z_check;
        Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1> z_check_var(dim);

        for (int i = 0; i < n_monte_carlo_; ++i) {
          // Draw from standard normal and transform to unconstrained space
          z_check = stan::prob::multi_normal_rng(zero_mean, eye, rng_);
          muL.to_unconstrained(z_check);

          stan::model::gradient(model_, z_check, tmp_lp, tmp_mu_grad, &std::cout);

          mu_grad += tmp_mu_grad;
        }
        mu_grad /= static_cast<double>(n_monte_carlo_);
      }



      void test() {
        if (out_stream_) *out_stream_ << "This is base_vb::bbvb::test()" << std::endl;

        if (out_stream_) *out_stream_ << "cont_params_ = " << std::endl
                                      << cont_params_ << std::endl << std::endl;


        // Init print out
        double lp = 0.0;
        double elbo = 0.0;
        Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model_.num_params_r());

        stan::model::gradient(model_, cont_params_, lp, init_grad, &std::cout);
        std::cout << "stan::model::gradient, lp = " << lp << std::endl;
        std::cout << "init_grad = " << init_grad << std::endl;


        // mu, L
        Eigen::VectorXd mu = cont_params_;
        Eigen::MatrixXd L  = Eigen::MatrixXd::Identity(model_.num_params_r(), model_.num_params_r());

        latent_vars muL = latent_vars(mu,L);


        // Eigen::VectorXd bla;

        // bla = Eigen::VectorXd::Constant(model_.num_params_r(),-10.0);
        // stan::model::gradient(model_, bla, lp, init_grad, &std::cout);
        // std::cout << "lp = " << lp << std::endl;
        // std::cout << "grad = " << init_grad << std::endl;
        // muL.set_mu(bla);
        // elbo = calc_ELBO(muL);
        // if (out_stream_) *out_stream_ << "elbo = " << elbo << std::endl << std::endl;

        // bla = Eigen::VectorXd::Constant(model_.num_params_r(),0.0);
        // stan::model::gradient(model_, bla, lp, init_grad, &std::cout);
        // std::cout << "lp = " << lp << std::endl;
        // std::cout << "grad = " << init_grad << std::endl;
        // muL.set_mu(bla);
        // elbo = calc_ELBO(muL);
        // if (out_stream_) *out_stream_ << "elbo = " << elbo << std::endl << std::endl;

        // bla = Eigen::VectorXd::Constant(model_.num_params_r(),2.7);
        // stan::model::gradient(model_, bla, lp, init_grad, &std::cout);
        // std::cout << "lp = " << lp << std::endl;
        // std::cout << "grad = " << init_grad << std::endl;
        // muL.set_mu(bla);
        // elbo = calc_ELBO(muL);
        // if (out_stream_) *out_stream_ << "elbo = " << elbo << std::endl << std::endl;

        // bla = Eigen::VectorXd::Constant(model_.num_params_r(),15.0);
        // stan::model::gradient(model_, bla, lp, init_grad, &std::cout);
        // std::cout << "lp = " << lp << std::endl;
        // std::cout << "grad = " << init_grad << std::endl;
        // muL.set_mu(bla);
        // elbo = calc_ELBO(muL);
        // if (out_stream_) *out_stream_ << "elbo = " << elbo << std::endl << std::endl;

        // bla = Eigen::VectorXd::Constant(model_.num_params_r(),100.0);
        // stan::model::gradient(model_, bla, lp, init_grad, &std::cout);
        // std::cout << "lp = " << lp << std::endl;
        // std::cout << "grad = " << init_grad << std::endl;
        // muL.set_mu(bla);
        // elbo = calc_ELBO(muL);
        // if (out_stream_) *out_stream_ << "elbo = " << elbo << std::endl << std::endl;


        Eigen::VectorXd mu_print;
        Eigen::VectorXd mu_grad = Eigen::VectorXd::Zero(model_.num_params_r());

        for (int i = 0; i < 20; ++i)
        {
          if (out_stream_) *out_stream_ << "---------------------" << std::endl
                                        << "  iter " << i << std::endl
                                        << "---------------------" << std::endl;

          calc_mu_grad(muL, mu_grad);

          if (out_stream_) *out_stream_ << "mu_grad = " << std::endl
                                        << mu_grad << std::endl;

          muL.set_mu(muL.mu() + 0.25*mu_grad);
          mu_print = muL.mu();

          if (out_stream_) *out_stream_ << "mu = " << std::endl
                                        << mu_print << std::endl;

          elbo = calc_ELBO(muL);
          if (out_stream_) *out_stream_ << "elbo = " << elbo << std::endl;

          // model_.template write_csv(rng_, mu_print, std::cout);

        }

        return;
      }

    protected:

      M& model_;
      Eigen::VectorXd cont_params_;
      BaseRNG& rng_;
      int n_monte_carlo_;

    };

  } // vb

} // stan

#endif

