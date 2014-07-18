#ifndef __STAN__VB__BASE_VB__HPP__
#define __STAN__VB__BASE_VB__HPP__

#include <ostream>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/autodiff.hpp>

#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <stan/prob/distributions/univariate/continuous/gamma.hpp>

#include <stan/prob/transform.hpp>

namespace stan {

  namespace vb {

    struct normal_functional {
      template <typename T>
      T operator()(Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const {
        return stan::prob::normal_log<false>(0, x, 1);
      }
    };

    struct gamma_functional {
     template <typename T>
     T operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> &x) const {
       T log_p(0);  // Will store the log probability + Jacobian
       T x_trans;
       x_trans = stan::prob::positive_constrain(x[0], log_p); // Transforms x to the positive scale, adds the Jacobian to log_p
       log_p += stan::prob::gamma_log<true>(5.0, x_trans, 1.0); // Evaluates the gamma density at the positive value
       return log_p; // Return the sum
     }
    };

    template <class M, class BaseRNG>
    class base_vb {

    public:

      base_vb(M& m, BaseRNG& rng, std::ostream* o, std::ostream* e):
      model_(m), rng_(rng), out_stream_(o), err_stream_(e) {

      };

      virtual ~base_vb() {};

      virtual void test(Eigen::VectorXd& input) {

        if (out_stream_) *out_stream_ << "This is base_vb::test" << std::endl;

        // Get constrained parameters list
        std::vector<std::string> constr_params;
        model_.constrained_param_names(constr_params);
        if (out_stream_) *out_stream_ << "constr_params[0]" << std::endl;
        if (out_stream_) *out_stream_ << constr_params[0] << std::endl;


        // Let's transform some random number, say 4.92
        std::vector<double> params_r;
        std::vector<int> params_i;
        std::vector<double> vars;

        params_r.push_back(4.92);
        params_i.push_back(1);

        model_.write_array(rng_, params_r, params_i, vars);
        if (out_stream_) *out_stream_ << "params_r[0]" << std::endl;
        if (out_stream_) *out_stream_ << params_r[0] << std::endl;
        if (out_stream_) *out_stream_ << "params_i[0]" << std::endl;
        if (out_stream_) *out_stream_ << params_i[0] << std::endl;
        if (out_stream_) *out_stream_ << "vars[0]" << std::endl;
        if (out_stream_) *out_stream_ << vars[0] << std::endl;

	// This is an addition Alp made from cmdstan's stan submodule

        // Let's invert that transformation
        // Eigen::Matrix<double,Eigen::Dynamic,1> params_r_out;
        // std::fstream data_stream(char(0), std::fstream::in);
        // stan::io::dump data_var_context(data_stream);
        // data_stream.close();

        // model_.transform_inits


        // Create "continuous parameters" input as a constant vector (for now)
        Eigen::VectorXd cont_params = Eigen::VectorXd::Constant(input.size(),1.35);
        if (out_stream_) *out_stream_ << "cont_params" << std::endl;
        if (out_stream_) *out_stream_ << cont_params << std::endl;

        // Let's compute a gradient of the Stan model
        double log_p;
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(cont_params.size());

        try {
          stan::model::gradient(model_, cont_params, log_p, grad, err_stream_);
        } catch (const std::exception& e) {
          this->_write_error_msg(err_stream_, e);
          log_p = - std::numeric_limits<double>::infinity();
        }

        if (out_stream_) *out_stream_ << "log_p" << std::endl;
        if (out_stream_) *out_stream_ << log_p << std::endl;
        if (out_stream_) *out_stream_ << "grad.transpose()" << std::endl;
        if (out_stream_) *out_stream_ << grad.transpose() << std::endl;

        try {
          stan::agrad::gradient(gamma_functional(), cont_params, log_p, grad);
        } catch (const std::exception& e) {
          this->_write_error_msg(err_stream_, e);
          log_p = - std::numeric_limits<double>::infinity();
        }

        if (out_stream_) *out_stream_ << "log_p" << std::endl;
        if (out_stream_) *out_stream_ << log_p << std::endl;
        if (out_stream_) *out_stream_ << "grad.transpose()" << std::endl;
        if (out_stream_) *out_stream_ << grad.transpose() << std::endl;


        // // We can do the same with an explicit distribution,
        // // although we don't quite have the same convenient
        // // wrappers to the autodiff library -- see the local
        // // functional defined above

        // // true is a template parameters specifying whether
        // // to compute constants of proportionality or not
        // stan::prob::normal_log<false>(cont_params, 0, 1);

        // try {
        //   stan::agrad::gradient(normal_functional(), cont_params, log_p, grad);
        // } catch (const std::exception& e) {
        //   this->_write_error_msg(err_stream_, e);
        //   log_p = - std::numeric_limits<double>::infinity();
        // }

        // if (out_stream_) *out_stream_ << grad.transpose() << std::endl;

        // // And let's generate a random variate for fun

        // double random_gaus = stan::prob::normal_rng(0, 1, rng_);


        // if (out_stream_) *out_stream_ << random_gaus << std::endl;

      }

    protected:

      M& model_;
      BaseRNG& rng_;

      std::ostream* out_stream_;
      std::ostream* err_stream_;

      void _write_error_msg(std::ostream* error_msgs,
                            const std::exception& e) {

        if (!error_msgs) return;

        *error_msgs << std::endl
                    << "Black Box Variational Bayes encountered an error:"
                    << std::endl
                    << e.what() << std::endl << std::endl;

      }

    };

  } // vb

} // stan

#endif

