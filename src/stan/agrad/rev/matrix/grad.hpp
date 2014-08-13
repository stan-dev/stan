#ifndef STAN__AGRAD__REV__MATRIX__GRAD_HPP
#define STAN__AGRAD__REV__MATRIX__GRAD_HPP


#include <stan/agrad/rev/matrix/Eigen_NumTraits.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/math/matrix/Eigen.hpp>

#include "Eigen/src/Core/DenseCoeffsBase.h"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/util/Constants.h"
#include "stan/agrad/rev/chainable.hpp"
#include "stan/agrad/rev/var_stack.hpp"
#include "stan/agrad/rev/vari.hpp"

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
