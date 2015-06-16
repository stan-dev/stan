#ifndef STAN_MATH_REV_MAT_FUNCTOR_GRADIENT_HPP
#define STAN_MATH_REV_MAT_FUNCTOR_GRADIENT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/core.hpp>

namespace stan {

  namespace math {

    using Eigen::Dynamic;

    /**
     * Calculate the value and the gradient of the specified function
     * at the specified argument.
     *
     * <p>The functor must implement
     *
     * <code>
     * stan::math::var
     * operator()(const
     * Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>&)
     * </code>
     *
     * using only operations that are defined for
     * <code>stan::math::var</code>.  This latter constraint usually
     * requires the functions to be defined in terms of the libraries
     * defined in Stan or in terms of functions with appropriately
     * general namespace imports that eventually depend on functions
     * defined in Stan.
     *
     * <p>Time and memory usage is on the order of the size of the
     * fully unfolded expression for the function applied to the
     * argument, independently of dimension.
     *
     * @tparam F Type of function
     * @param[in] f Function
     * @param[in] x Argument to function
     * @param[out] fx Function applied to argument
     * @param[out] grad_fx Gradient of function at argument
     */
    template <typename F>
    void
    gradient(const F& f,
             const Eigen::Matrix<double, Dynamic, 1>& x,
             double& fx,
             Eigen::Matrix<double, Dynamic, 1>& grad_fx) {
      using stan::math::var;
      start_nested();
      try {
        Eigen::Matrix<var, Dynamic, 1> x_var(x.size());
        for (int i = 0; i < x.size(); ++i)
          x_var(i) = x(i);
        var fx_var = f(x_var);
        fx = fx_var.val();
        grad_fx.resize(x.size());
        stan::math::grad(fx_var.vi_);
        for (int i = 0; i < x.size(); ++i)
          grad_fx(i) = x_var(i).adj();
      } catch (const std::exception& /*e*/) {
        stan::math::recover_memory_nested();
        throw;
      }
      stan::math::recover_memory_nested();
    }
  }  // namespace math
}  // namespace stan
#endif
