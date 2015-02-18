#ifndef STAN__MATH__MIX__MAT__FUNCTOR__DERIVATIVE_HPP
#define STAN__MATH__MIX__MAT__FUNCTOR__DERIVATIVE_HPP

#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/core/grad.hpp>
#include <stan/math/rev/core/set_zero_all_adjoints.hpp>
#include <vector>

namespace stan {

  namespace agrad {

    using Eigen::Dynamic;

    /**
     * Return the derivative of the specified univariate function at
     * the specified argument.
     *
     * @tparam T Argument type
     * @tparam F Function type
     * @param[in] f Function
     * @param[in] x Argument
     * @param[out] fx Value of function applied to argument
     * @param[out] dfx_dx Value of derivative
     */
    template <typename T, typename F>
    void
    derivative(const F& f,
               const T& x,
               T& fx,
               T& dfx_dx)  {
      fvar<T> x_fvar = fvar<T>(x, 1.0);
      fvar<T> fx_fvar = f(x_fvar);
      fx = fx_fvar.val_;
      dfx_dx = fx_fvar.d_;
    }

  }  // namespace agrad
}  // namespace stan
#endif
