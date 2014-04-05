#ifndef __STAN__VB__BASE_VB__HPP__
#define __STAN__VB__BASE_VB__HPP__

#include <ostream>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/autodiff.hpp>

#include <stan/prob/distributions/univariate/continuous/normal.hpp>

namespace stan {

  namespace vb {
    
    struct normal_functional {
      template <typename T>
      T operator()(Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const {
        return stan::prob::normal_log<false>(x, 0, 1);
      }
    };
    
    template <class M, class BaseRNG>
    class base_vb {
      
    public:
      
      base_vb(M& m, BaseRNG& rng, std::ostream* o, std::ostream* e):
      model_(m), rng_(rng), out_stream_(o), err_stream_(e) {};
      
      virtual ~base_vb() {};
      
      virtual void test(Eigen::VectorXd& q) {
        
        // Let's compute a gradient of the Stan model
        double log_p;
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(q.size());
        
        try {
          stan::model::gradient(model_, q, log_p, grad, err_stream_);
        } catch (const std::exception& e) {
          this->_write_error_msg(err_stream_, e);
          log_p = - std::numeric_limits<double>::infinity();
        }
        
        if (out_stream_) *out_stream_ << grad.transpose() << std::endl;
        
        // We can do the same with an explicit distribution,
        // although we don't quite have the same convenient
        // wrappers to the autodiff library -- see the local
        // functional defined above
        
        // true is a template parameters specifying whether
        // to compute constants of proportionality or not
        stan::prob::normal_log<false>(q, 0, 1);
        
        try {
          stan::agrad::gradient(normal_functional(), q, log_p, grad);
        } catch (const std::exception& e) {
          this->_write_error_msg(err_stream_, e);
          log_p = - std::numeric_limits<double>::infinity();
        }
        
        if (out_stream_) *out_stream_ << grad.transpose() << std::endl;
        
        // And let's generate a random variate for fun

        double random_gaus = stan::prob::normal_rng(0, 1, rng_);
        
        
        if (out_stream_) *out_stream_ << random_gaus << std::endl;
        
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

