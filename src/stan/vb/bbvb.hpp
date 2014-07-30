#ifndef STAN__VB__BBVB__HPP
#define STAN__VB__BBVB__HPP

#include <ostream>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/log_determinant.hpp>
#include <stan/model/util.hpp>

#include <stan/math/functions.hpp>  // I had to add these two lines beceause
#include <stan/math/matrix.hpp>     // bbvb_test.cpp wouldn't compile...

#include <stan/prob/distributions/multivariate/continuous/multi_normal.hpp>

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

      void test() {
        if (out_stream_) *out_stream_ << "This is base_vb::bbvb::test()" << std::endl;

        if (out_stream_) *out_stream_ << "cont_params_ = " << std::endl
                                      << cont_params_ << std::endl << std::endl;

        // OK, let's use this gradient thing to compute the log_prob
        double lp(0.0);
        Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model_.num_params_r());

        stan::model::gradient(model_, cont_params_, lp, init_grad, &std::cout);
        std::cout << "stan::model::gradient, lp = " << lp << std::endl;

        std::cout << "DOUBLE" << std::endl;

        std::cout << "<true,true> lp = "
                  << model_.template log_prob<true,true>(cont_params_, &std::cout)
                  << std::endl;
        std::cout << "<true,false> lp = "
                  << model_.template log_prob<true,false>(cont_params_, &std::cout)
                  << std::endl;
        std::cout << "<false,true> lp = "
                  << model_.template log_prob<false,true>(cont_params_, &std::cout)
                  << std::endl;
        std::cout << "<false,false> lp = "
                  << model_.template log_prob<false,false>(cont_params_, &std::cout)
                  << std::endl;

        std::cout << "VAR" << std::endl;

        Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1> cont_params_var(1);
        cont_params_var << cont_params_(0);

        std::cout << "<true,true> lp = "
                  << model_.template log_prob<true,true>(cont_params_var, &std::cout)
                  << std::endl;
        std::cout << "<true,false> lp = "
                  << model_.template log_prob<true,false>(cont_params_var, &std::cout)
                  << std::endl;
        std::cout << "<false,true> lp = "
                  << model_.template log_prob<false,true>(cont_params_var, &std::cout)
                  << std::endl;
        std::cout << "<false,false> lp = "
                  << model_.template log_prob<true,true>(cont_params_var, &std::cout)
                  << std::endl;



        // Now let's test this latent_vars class we just wrote
        Eigen::VectorXd mu = Eigen::VectorXd::Constant(model_.num_params_r(), 5.0);
        Eigen::MatrixXd L  = Eigen::MatrixXd::Identity(model_.num_params_r(), model_.num_params_r());

        latent_vars muL = latent_vars(mu,L);

        if (out_stream_) *out_stream_ << "muL.mu() = " << std::endl
                                      << muL.mu() << std::endl;

        if (out_stream_) *out_stream_ << "muL.L() = " << std::endl
                                      << muL.L() << std::endl;

        Eigen::VectorXd x = Eigen::VectorXd::Constant(model_.num_params_r(),10.0);

        if (out_stream_) *out_stream_ << "x = " << std::endl
                                      << x << std::endl;

        muL.to_unconstrained(x);
        if (out_stream_) *out_stream_ << "unconstrained x = " << std::endl
                                      << x << std::endl;

        muL.to_standardized(x);
        if (out_stream_) *out_stream_ << "standardized x = " << std::endl
                                      << x << std::endl;


        // Now let's call this ELBO function
        double elbo = calc_ELBO(muL);

        if (out_stream_) *out_stream_ << "elbo = " << std::endl
                                      << elbo << std::endl << std::endl;

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

