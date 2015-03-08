#ifndef STAN__MATH__PRIM__MAT__FUNCTOR__FINITE_DIFF_GRAD_HESSIAN_HPP
#define STAN__MATH__PRIM__MAT__FUNCTOR__FINITE_DIFF_GRAD_HESSIAN_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/mix/mat/functor/gradient.hpp>

namespace stan {
  
  namespace agrad {

    /** 
     * Calculate the value and the gradient of the hessian of the specified
     * function at the specified argument using second-order autodiff and
       * first-order finite difference.  
     *
     * <p>The functor must implement 
     * 
     * <code>
     * double
     * operator()(const
     * Eigen::Matrix<double,Eigen::Dynamic,1>&)
     * </code>
     *
     * @tparam F Type of function
     * @param[in] f Function
     * @param[in] x Argument to function
     * @param[out] fx Function applied to argument
     * @param[out] grad_hess_fx gradient of Hessian of function at argument
     * @param[in] epsilon perturbation size
     */
    template <typename F>
    void
    finite_diff_grad_hessian_auto(const F& f,
                                  const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
                                  double& fx,
                                  std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >& grad_hess_fx, 
                                  const double epsilon = 1e-04) {
      using Eigen::Matrix;
      using Eigen::Dynamic;

      int d = x.size();
      double dummy_fx_eval;

      Matrix<double,Dynamic,1> x_temp(x);
      Matrix<double,Dynamic,1> grad_auto(d);
      Matrix<double,Dynamic,Dynamic> H_auto(d,d);
      Matrix<double,Dynamic,Dynamic> H_diff(d,d);

      
      for (int i = 0; i < d; ++i){
        H_diff.setZero();

        x_temp(i) += 2.0 * epsilon;
        hessian(f, x_temp, dummy_fx_eval, grad_auto, H_auto);
        H_diff = -H_auto;

        x_temp(i) = x(i) + -2.0 * epsilon;
        hessian(f, x_temp, dummy_fx_eval, grad_auto, H_auto);
        H_diff += H_auto;

        x_temp(i) = x(i) + epsilon;
        hessian(f, x_temp, dummy_fx_eval, grad_auto, H_auto);
        H_diff += 8.0 * H_auto;

        x_temp(i) = x(i) + -epsilon;
        hessian(f, x_temp, dummy_fx_eval, grad_auto, H_auto);
        H_diff -= 8.0 * H_auto;

        x_temp(i) = x(i);
        H_diff /= 12.0 * epsilon;
        
        grad_hess_fx.push_back(H_diff);
      }
      fx = f(x);
    }

  }
}
#endif
