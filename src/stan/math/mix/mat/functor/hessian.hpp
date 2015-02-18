#ifndef STAN__MATH__MIX__MAT__FUNCTOR__HESSIAN_HPP
#define STAN__MATH__MIX__MAT__FUNCTOR__HESSIAN_HPP

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
     * Calculate the value, the gradient, and the Hessian,
     * of the specified function at the specified argument in
     * O(N^2) time and O(N^2) space.
     *
     * <p>The functor must implement
     *
     * <code>
     * stan::agrad::fvar<stan::agrad::var>
     * operator()(const
     * Eigen::Matrix<stan::agrad::fvar<stan::agrad::var>, Eigen::Dynamic, 1>&)
     * </code>
     *
     * using only operations that are defined for
     * <code>stan::agrad::fvar</code> and <code>stan::agrad::var</code>.
     *
     * This latter constraint usually
     * requires the functions to be defined in terms of the libraries
     * defined in Stan or in terms of functions with appropriately
     * general namespace imports that eventually depend on functions
     * defined in Stan.
     *
     * @tparam F Type of function
     * @param[in] f Function
     * @param[in] x Argument to function
     * @param[out] fx Function applied to argument
     * @param[out] grad gradient of function at argument
     * @param[out] H Hessian of function at argument
     */
    template <typename F>
    void
    hessian(const F& f,
            const Eigen::Matrix<double, Dynamic, 1>& x,
            double& fx,
            Eigen::Matrix<double, Dynamic, 1>& grad,
            Eigen::Matrix<double, Dynamic, Dynamic>& H) {
      H.resize(x.size(), x.size());
      grad.resize(x.size());
      try {
        for (int i = 0; i < x.size(); ++i) {
          start_nested();
          Eigen::Matrix<fvar<var>, Dynamic, 1> x_fvar(x.size());
          for (int j = 0; j < x.size(); ++j)
            x_fvar(j) = fvar<var>(x(j), i == j);
          fvar<var> fx_fvar = f(x_fvar);
          grad(i) = fx_fvar.d_.val();
          if (i == 0) fx = fx_fvar.val_.val();
          stan::agrad::grad(fx_fvar.d_.vi_);
          for (int j = 0; j < x.size(); ++j)
            H(i, j) = x_fvar(j).val_.adj();
          stan::agrad::recover_memory_nested();
        }
      } catch (const std::exception& e) {
        stan::agrad::recover_memory_nested();
        throw;
      }
    }
    // time O(N^3);  space O(N^2)
    template <typename T, typename F>
    void
    hessian(const F& f,
            const Eigen::Matrix<T, Dynamic, 1>& x,
            T& fx,
            Eigen::Matrix<T, Dynamic, 1>& grad,
            Eigen::Matrix<T, Dynamic, Dynamic>& H) {
      H.resize(x.size(), x.size());
      grad.resize(x.size());
      Eigen::Matrix<fvar<fvar<T> >, Dynamic, 1> x_fvar(x.size());
      for (int i = 0; i < x.size(); ++i) {
        for (int j = i; j < x.size(); ++j) {
          for (int k = 0; k < x.size(); ++k)
            x_fvar(k) = fvar<fvar<T> >(fvar<T>(x(k), j == k),
                                       fvar<T>(i == k, 0));
          fvar<fvar<T> > fx_fvar = f(x_fvar);
          if (j == 0)
            fx = fx_fvar.val_.val_;
          if (i == j)
            grad(i) = fx_fvar.d_.val_;
          H(i, j) = fx_fvar.d_.d_;
          H(j, i) = H(i, j);
        }
      }
    }

  }  // namespace agrad
}  // namespace stan
#endif
