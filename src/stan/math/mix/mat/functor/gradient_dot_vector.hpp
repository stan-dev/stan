#ifndef STAN__MATH__MIX__MAT__FUNCTOR__GRADIENT_DOT_VECTOR_HPP
#define STAN__MATH__MIX__MAT__FUNCTOR__GRADIENT_DOT_VECTOR_HPP

#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/core/grad.hpp>
#include <stan/math/rev/core/set_zero_all_adjoints.hpp>
#include <vector>

namespace stan {

  namespace agrad {

    using Eigen::Dynamic;

    // aka directional derivative (not length normalized)
    // T2 must be assignable to T1
    template <typename T1, typename T2, typename F>
    void
    gradient_dot_vector(const F& f,
                        const Eigen::Matrix<T1, Dynamic, 1>& x,
                        const Eigen::Matrix<T2, Dynamic, 1>& v,
                        T1& fx,
                        T1& grad_fx_dot_v) {
      using stan::agrad::fvar;
      using stan::agrad::var;
      using Eigen::Matrix;
      Matrix<fvar<T1>, Dynamic, 1> x_fvar(x.size());
      for (int i = 0; i < x.size(); ++i)
        x_fvar(i) = fvar<T1>(x(i), v(i));
      fvar<T1> fx_fvar = f(x_fvar);
      fx = fx_fvar.val_;
      grad_fx_dot_v = fx_fvar.d_;
    }

  }  // namespace agrad
}  // namespace stan
#endif
