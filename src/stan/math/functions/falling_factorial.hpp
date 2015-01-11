#ifndef STAN__MATH__FUNCTIONS__FALLING_FACTORIAL_HPP
#define STAN__MATH__FUNCTIONS__FALLING_FACTORIAL_HPP

#include <stan/math/functions/log_falling_factorial.hpp>

namespace stan {
  namespace math {

    /**
     *
       \f[
       \mbox{falling\_factorial}(x,n) = 
       \begin{cases}
         \textrm{error} & \mbox{if } x \leq 0\\
         (x)_n & \mbox{if } x > 0 \textrm{ and } -\infty \leq n \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } n = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{falling\_factorial}(x,n)}{\partial x} = 
       \begin{cases}
         \textrm{error} & \mbox{if } x \leq 0\\
         \frac{\partial\, (x)_n}{\partial x} & \mbox{if } x > 0 \textrm{ and } -\infty \leq n \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } n = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{falling\_factorial}(x,n)}{\partial n} = 
       \begin{cases}
         \textrm{error} & \mbox{if } x \leq 0\\
         \frac{\partial\, (x)_n}{\partial n} & \mbox{if } x > 0 \textrm{ and } -\infty \leq n \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } n = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       (x)_n=\frac{\Gamma(x+1)}{\Gamma(x-n+1)}
       \f]
       
       \f[
       \frac{\partial \, (x)_n}{\partial x} = (x)_n\Psi(x+1)
       \f]
       
       \f[
       \frac{\partial \, (x)_n}{\partial n} = -(x)_n\Psi(n+1)
       \f]
     *
     */
    template<typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type
    falling_factorial(const T1 x, const T2 n) { 
      return std::exp(stan::math::log_falling_factorial(x,n)); 
    }

  }
}

#endif
