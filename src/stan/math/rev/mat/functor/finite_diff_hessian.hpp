#ifndef STAN_MATH_REV_MAT_FUNCTOR_FINITE_DIFF_HESSIAN_HPP
#define STAN_MATH_REV_MAT_FUNCTOR_FINITE_DIFF_HESSIAN_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/functor/gradient.hpp>

namespace stan {

  namespace math {

    /**
     * Calculate the value and the hessian of the specified function
     * at the specified argument using first-order autodiff and
     * first-order finite difference.
     *
     * <p>The functor must implement
     *
     * <code>
     * double
     * operator()(const
     * Eigen::Matrix<double, Eigen::Dynamic, 1>&)
     * </code>
     *
     * The reference for this algorithm is:
     *
     * De Levie: An improved numerical approximation
     * for the first derivative, page 3
     *
     * This function involves 4 calls to f. Error
     * on order of epsilon ^ 4.
     *
     * @tparam F Type of function
     * @param[in] f Function
     * @param[in] x Argument to function
     * @param[out] fx Function applied to argument
     * @param[out] hess_fx Hessian of function at argument
     * @param[in] epsilon perturbation size
     */
    template <typename F>
    void
    finite_diff_hessian_autodiff(const F& f,
                                 const Eigen::Matrix<double, -1, 1>& x,
                                 double& fx,
                                 Eigen::Matrix<double, -1, 1>& grad_fx,
                                 Eigen::Matrix<double, -1, -1>& hess_fx,
                                 const double epsilon = 1e-03) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using Eigen::VectorXd;

      int d = x.size();
      double dummy_fx_eval;

      Matrix<double, Dynamic, 1> x_temp(x);
      Matrix<double, Dynamic, 1> g_auto(d);
      hess_fx.resize(d, d);


      gradient(f, x, fx, grad_fx);
      for (int i = 0; i < d; ++i) {
        VectorXd g_diff = VectorXd::Zero(d);
        x_temp(i) = x(i) + 2.0 * epsilon;
        gradient(f, x_temp, dummy_fx_eval, g_auto);
        g_diff = -g_auto;

        x_temp(i) = x(i) + -2.0 * epsilon;
        gradient(f, x_temp, dummy_fx_eval, g_auto);
        g_diff += g_auto;

        x_temp(i) = x(i) + epsilon;
        gradient(f, x_temp, dummy_fx_eval, g_auto);
        g_diff += 8.0 * g_auto;

        x_temp(i) = x(i) - epsilon;
        gradient(f, x_temp, dummy_fx_eval, g_auto);
        g_diff -= 8.0 * g_auto;

        x_temp(i) = x(i);
        g_diff /= 12.0 * epsilon;

        hess_fx.col(i) = g_diff;
      }
    }
  }
}
#endif
