#ifndef STAN__PROB__WELFORD__VAR__ESTIMATOR__BETA
#define STAN__PROB__WELFORD__VAR__ESTIMATOR__BETA

#include <stan/math/matrix/Eigen.hpp>
#include <new>
#include <vector>

#include "Eigen/src/Core/../plugins/CommonCwiseBinaryOps.h"
#include "Eigen/src/Core/../plugins/CommonCwiseUnaryOps.h"
#include "Eigen/src/Core/../plugins/MatrixCwiseBinaryOps.h"
#include "Eigen/src/Core/Assign.h"
#include "Eigen/src/Core/CwiseBinaryOp.h"
#include "Eigen/src/Core/CwiseNullaryOp.h"
#include "Eigen/src/Core/CwiseUnaryOp.h"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/MatrixBase.h"
#include "Eigen/src/Core/Transpose.h"
#include "Eigen/src/Core/util/Memory.h"

namespace stan {
  
  namespace prob {
    
    class welford_var_estimator {
      
    public:
      
      welford_var_estimator(int n): _m(Eigen::VectorXd::Zero(n)),
                                    _m2(Eigen::VectorXd::Zero(n))
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
        _m2 += delta.cwiseProduct(q - _m);
        
      }
      
      int num_samples() { return _num_samples; }
      
      void sample_mean(Eigen::VectorXd& mean) { mean = _m; }
      
      void sample_variance(Eigen::VectorXd& var) {
        if(_num_samples > 1)
          var = _m2 / (_num_samples - 1.0);
      }
      
    protected:
      
      double _num_samples;
      
      Eigen::VectorXd _m;
      Eigen::VectorXd _m2;
      
    };
    
  } // prob
  
} // stan


#endif
