#ifndef STAN__AGRAD__REV__MATRIX__GRAD_HPP
#define STAN__AGRAD__REV__MATRIX__GRAD_HPP


#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/rev/matrix/Eigen_NumTraits.hpp>
#include <stan/agrad/rev/var.hpp>

namespace stan {

  namespace agrad {
   
    void grad(var& v,
              Eigen::Matrix<var,Eigen::Dynamic,1>& x,
              Eigen::VectorXd& g) {
      stan::agrad::grad(v.vi_);
      g.resize(x.size());
      for (int i = 0; i < x.size(); ++i)
        g(i) = x(i).vi_->adj_;
      recover_memory();
    }
    
  }
}

#endif
