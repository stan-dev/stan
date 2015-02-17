#ifndef STAN__MATH__PRIM__MAT__FUN__WELFORD_COVAR_ESTIMATOR_HPP
#define STAN__MATH__PRIM__MAT__FUN__WELFORD_COVAR_ESTIMATOR_HPP

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  
  namespace prob {
    
    class welford_covar_estimator {
      
    public:
      
      welford_covar_estimator(int n): _m(Eigen::VectorXd::Zero(n)),
                                      _m2(Eigen::MatrixXd::Zero(n, n))
      { restart(); }
      
      void restart() {
        _num_samples = 0;
        _m.setZero();
        _m2.setZero();
      }
      
      void add_sample(const Eigen::VectorXd& q) {
        
        ++_num_samples;
        
        Eigen::VectorXd delta(q - _m);
        _m  += delta / _num_samples;
        _m2 += (q - _m) * delta.transpose();
        
      }
      
      int num_samples() { return _num_samples; }
      
      void sample_mean(Eigen::VectorXd& mean) { mean = _m; }
      
      void sample_covariance(Eigen::MatrixXd& covar) {
        if(_num_samples > 1)
          covar = _m2 / (_num_samples - 1.0);
      }
      
    protected:
      
      double _num_samples;
      
      Eigen::VectorXd _m;
      Eigen::MatrixXd _m2;
      
    };
    
  } // prob
  
} // stan


#endif
