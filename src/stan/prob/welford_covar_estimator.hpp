#ifndef __STAN__PROB__WELFORD__COVAR__ESTIMATOR__BETA__
#define __STAN__PROB__WELFORD__COVAR__ESTIMATOR__BETA__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

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
      
      void add_sample(std::vector<double>& q) {
        
        Eigen::Map<Eigen::VectorXd> x(&q[0], q.size());
        
        ++_num_samples;
        
        Eigen::VectorXd delta(x - _m);
        _m  += delta / _num_samples;
        _m2 += (x - _m) * delta.transpose();
        
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