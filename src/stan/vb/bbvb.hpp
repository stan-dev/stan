#ifndef __STAN__VB__BBVB__HPP__
#define __STAN__VB__BBVB__HPP__

#include <ostream>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/log_determinant.hpp>
#include <stan/model/util.hpp>

#include <stan/prob/distributions/multivariate/continuous/multi_normal.hpp>

#include <stan/vb/base_vb.hpp>
#include <stan/vb/latent_vars.hpp>

namespace stan {

  namespace vb {

    template <class M, class BaseRNG>
    class bbvb : public base_vb
    {

    public:

      bbvb(M& m,
           Eigen::VectorXd const& cont_params,
           BaseRNG& rng,
           std::ostream* o, std::ostream* e):
        base_vb(o, e),
        model_(m),
        cont_params_(cont_params),
        rng_(rng),
        num_of_Monte_Carlo_(100) {};

      virtual ~bbvb() {};

      /**
       *  Calculates the "blackbox" Evidence Lower BOund (ELBO) by sampling
       *  from the standard multivariate normal (for now), affine transforming
       *  the sample, and evaluating the log joint, adjusted by the entropy
       *  term of the normal, which is proportional to 0.5*logdet(L^T L)
       **/
      double calc_ELBO(latent_vars const& muL)
      {
        double elbo(0);
        int dim = muL.dimension();
        Eigen::VectorXd zero_mean = Eigen::VectorXd::Zero(dim);
        Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(dim, dim);

        Eigen::MatrixXd LTL = muL.L().transpose()*muL.L();

        Eigen::VectorXd z_check;
        for (int i = 0; i < num_of_Monte_Carlo_; ++i)
        {
          z_check = stan::prob::multi_normal_rng(zero_mean, eye, rng_);
          muL.to_unconstrained(z_check);
          elbo += model_.template
                    log_prob<false, true>(z_check,
                                          &std::cout);
        }
        elbo /= num_of_Monte_Carlo_;

        elbo += 0.5*(stan::math::log_determinant(LTL));

        return elbo;
      };

      void test()
      {
        if (out_stream_) *out_stream_ << "This is base_vb::bbvb::test()" << std::endl;

        // OK, let's use this gradient thing to compute the log_prob
        double init_log_prob;
        Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model_.num_params_r());

        try {
          stan::model::gradient(model_, cont_params_, init_log_prob, init_grad, &std::cout);
        } catch (const std::exception& e) {
          std::cout << "Rejecting initialization at zero because of gradient failure."
                    << std::endl
                    << e.what() << std::endl;
          return;
        }

        // Print some stuff out
        if (out_stream_) *out_stream_ << "cont_params_ = " << std::endl
                                      << cont_params_ << std::endl << std::endl;

        if (out_stream_) *out_stream_ << "init_log_prob = " << std::endl
                                      << init_log_prob << std::endl << std::endl;

        if (out_stream_) *out_stream_ << "init_grad = " << std::endl
                                      << init_grad << std::endl << std::endl;


        // I guess this is the one I should use to compute the ELBO
        double lp(0);
        try {
          lp = model_.template log_prob<false, true>(cont_params_, &std::cout);
        } catch (const std::exception& e) {
          lp = -std::numeric_limits<double>::infinity();
          return;
        }

        if (out_stream_) *out_stream_ << "lp = " << std::endl
                                      << lp << std::endl << std::endl;


        // Now let's test this latent_vars class we just wrote
        Eigen::VectorXd mu = Eigen::VectorXd::Constant(model_.num_params_r(),5.0);
        Eigen::MatrixXd L = Eigen::MatrixXd::Identity(model_.num_params_r(),model_.num_params_r());

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
      };

    protected:

      M& model_;
      Eigen::VectorXd cont_params_;
      BaseRNG& rng_;
      int num_of_Monte_Carlo_;

      void write_error_msg_(std::ostream* error_msgs, const std::exception& e)
      {
        if (!error_msgs) return;

        *error_msgs << std::endl
                    << "[stan::vb::bbvb.hpp] encountered an error:"
                    << std::endl
                    << e.what() << std::endl << std::endl;
      };

    };

  } // vb

} // stan

#endif

