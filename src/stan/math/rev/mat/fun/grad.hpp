#ifndef STAN__MATH__REV__MAT__FUN__GRAD_HPP
#define STAN__MATH__REV__MAT__FUN__GRAD_HPP


#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/rev/arr/meta/var.hpp>
#include <stan/math/rev/arr/meta/grad.hpp>

namespace stan {

  namespace agrad {
   
    /**
     * Propagate chain rule to calculate gradients starting from
     * the specified variable.  Resizes the input vector to be the
     * correct size.
     *
     * The grad() function does not itself recover any memory.  use
     * <code>agrad::recover_memory()</code> or
     * <code>agrad::recover_memory_nested()</code>, defined in ,
     * defined in agrad/rev/var_stack.hpp, to recover memory.
     *
     * @param[in] v Value of function being differentiated
     * @param[in] x Variables being differentiated with respect to
     * @param[out] g Gradient, d/dx v, evaluated at x.
     */
    void grad(var& v,
              Eigen::Matrix<var,Eigen::Dynamic,1>& x,
              Eigen::VectorXd& g) {
      stan::agrad::grad(v.vi_);
      g.resize(x.size());
      for (int i = 0; i < x.size(); ++i)
        g(i) = x(i).vi_->adj_;
    }
    
  }
}

#endif
