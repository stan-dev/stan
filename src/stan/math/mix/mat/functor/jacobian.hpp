#ifndef STAN__MATH__MIX__MAT__FUNCTOR__JACOBIAN_HPP
#define STAN__MATH__MIX__MAT__FUNCTOR__JACOBIAN_HPP

#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/core/grad.hpp>
#include <stan/math/rev/core/set_zero_all_adjoints.hpp>
#include <stan/math/rev/core/recover_memory_nested.hpp>
#include <stan/math/rev/core/start_nested.hpp>
#include <vector>

namespace stan {

  namespace agrad {

    using Eigen::Dynamic;


    template <typename F>
    void
    jacobian(const F& f,
             const Eigen::Matrix<double, Dynamic, 1>& x,
             Eigen::Matrix<double, Dynamic, 1>& fx,
             Eigen::Matrix<double, Dynamic, Dynamic>& J) {
      using Eigen::Matrix;
      using stan::agrad::var;
      start_nested();
      try {
        Matrix<var, Dynamic, 1> x_var(x.size());
        for (int k = 0; k < x.size(); ++k)
          x_var(k) = x(k);
        Matrix<var, Dynamic, 1> fx_var = f(x_var);
        fx.resize(fx_var.size());
        for (int i = 0; i < fx_var.size(); ++i)
          fx(i) = fx_var(i).val();
        J.resize(x.size(), fx_var.size());
        for (int i = 0; i < fx_var.size(); ++i) {
          if (i > 0)
            set_zero_all_adjoints();
          grad(fx_var(i).vi_);
          for (int k = 0; k < x.size(); ++k)
            J(k, i) = x_var(k).adj();
        }
      } catch (const std::exception& e) {
        stan::agrad::recover_memory_nested();
        throw;
      }
      stan::agrad::recover_memory_nested();
    }
    template <typename T, typename F>
    void
    jacobian(const F& f,
             const Eigen::Matrix<T, Dynamic, 1>& x,
             Eigen::Matrix<T, Dynamic, 1>& fx,
             Eigen::Matrix<T, Dynamic, Dynamic>& J) {
      using Eigen::Matrix;
      using stan::agrad::fvar;
      Matrix<fvar<T>, Dynamic, 1> x_fvar(x.size());
      for (int i = 0; i < x.size(); ++i) {
        for (int k = 0; k < x.size(); ++k)
          x_fvar(k) = fvar<T>(x(k), i == k);
        Matrix<fvar<T>, Dynamic, 1> fx_fvar
          = f(x_fvar);
        if (i == 0) {
          J.resize(x.size(), fx_fvar.size());
          fx.resize(fx_fvar.size());
          for (int k = 0; k < fx_fvar.size(); ++k)
            fx(k) = fx_fvar(k).val_;
        }
        for (int k = 0; k < fx_fvar.size(); ++k) {
          J(i, k) = fx_fvar(k).d_;
        }
      }
    }

  }  // namespace agrad
}  // namespace stan
#endif
