#ifndef STAN_MATH_MIX_MAT_FUNCTOR_HESSIAN_TIMES_VECTOR_HPP
#define STAN_MATH_MIX_MAT_FUNCTOR_HESSIAN_TIMES_VECTOR_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/core.hpp>
#include <vector>

namespace stan {

  namespace math {

    using Eigen::Dynamic;

    template <typename F>
    void
    hessian_times_vector(const F& f,
                         const Eigen::Matrix<double, Dynamic, 1>& x,
                         const Eigen::Matrix<double, Dynamic, 1>& v,
                         double& fx,
                         Eigen::Matrix<double, Dynamic, 1>& Hv) {
      using stan::math::fvar;
      using stan::math::var;
      using Eigen::Matrix;
      start_nested();
      try {
        Matrix<var, Dynamic, 1> x_var(x.size());
        for (int i = 0; i < x_var.size(); ++i)
          x_var(i) = x(i);
        var fx_var;
        var grad_fx_var_dot_v;
        gradient_dot_vector(f, x_var, v, fx_var, grad_fx_var_dot_v);
        fx = fx_var.val();
        stan::math::grad(grad_fx_var_dot_v.vi_);
        Hv.resize(x.size());
        for (int i = 0; i < x.size(); ++i)
          Hv(i) = x_var(i).adj();
      } catch (const std::exception& e) {
        stan::math::recover_memory_nested();
        throw;
      }
      stan::math::recover_memory_nested();
    }
    template <typename T, typename F>
    void
    hessian_times_vector(const F& f,
                         const Eigen::Matrix<T, Dynamic, 1>& x,
                         const Eigen::Matrix<T, Dynamic, 1>& v,
                         T& fx,
                         Eigen::Matrix<T, Dynamic, 1>& Hv) {
      using Eigen::Matrix;
      Matrix<T, Dynamic, 1> grad;
      Matrix<T, Dynamic, Dynamic> H;
      hessian(f, x, fx, grad, H);
      Hv = H * v;
    }

  }  // namespace math
}  // namespace stan
#endif
