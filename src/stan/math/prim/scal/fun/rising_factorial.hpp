#ifndef STAN_MATH_PRIM_SCAL_FUN_RISING_FACTORIAL_HPP
#define STAN_MATH_PRIM_SCAL_FUN_RISING_FACTORIAL_HPP

#include <stan/math/prim/scal/fun/log_rising_factorial.hpp>
#include <cmath>

namespace stan {
  namespace math {

    /**
     *
       \f[
       \mbox{rising\_factorial}(x, n) =
       \begin{cases}
         \textrm{error} & \mbox{if } x \leq 0\\
         x^{(n)} & \mbox{if } x > 0 \textrm{ and } -\infty \leq n \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } n = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\, \mbox{rising\_factorial}(x, n)}{\partial x} =
       \begin{cases}
         \textrm{error} & \mbox{if } x \leq 0\\
         \frac{\partial\, x^{(n)}}{\partial x} & \mbox{if } x > 0 \textrm{ and } -\infty \leq n \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } n = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\, \mbox{rising\_factorial}(x, n)}{\partial n} =
       \begin{cases}
         \textrm{error} & \mbox{if } x \leq 0\\
         \frac{\partial\, x^{(n)}}{\partial n} & \mbox{if } x > 0 \textrm{ and } -\infty \leq n \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } n = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       x^{(n)}=\frac{\Gamma(x+n)}{\Gamma(x)}
       \f]

       \f[
       \frac{\partial \, x^{(n)}}{\partial x} = x^{(n)}(\Psi(x+n)-\Psi(x))
       \f]

       \f[
       \frac{\partial \, x^{(n)}}{\partial n} = (x)_n\Psi(x+n)
       \f]
     *
     */
    template<typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1, T2>::type
    rising_factorial(const T1 x, const T2 n) {
      using std::exp;
      return exp(stan::math::log_rising_factorial(x, n));
    }

  }
}

#endif
