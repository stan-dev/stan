#ifndef STAN__MATH__FUNCTIONS__LOG_FALLING_FACTORIAL_HPP
#define STAN__MATH__FUNCTIONS__LOG_FALLING_FACTORIAL_HPP

#include <boost/math/special_functions/gamma.hpp>

namespace stan {
  namespace math {
    
    /**
     *
       \f[
       \mbox{log\_falling\_factorial}(x,n) = 
       \begin{cases}
         \textrm{error} & \mbox{if } x \leq 0\\
         \ln (x)_n & \mbox{if } x > 0 \textrm{ and } -\infty \leq n \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } n = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{log\_falling\_factorial}(x,n)}{\partial x} = 
       \begin{cases}
         \textrm{error} & \mbox{if } x \leq 0\\
         \Psi(x) & \mbox{if } x > 0 \textrm{ and } -\infty \leq n \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } n = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{log\_falling\_factorial}(x,n)}{\partial n} = 
       \begin{cases}
         \textrm{error} & \mbox{if } x \leq 0\\
         -\Psi(n) & \mbox{if } x > 0 \textrm{ and } -\infty \leq n \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } n = \textrm{NaN}
       \end{cases}
       \f]
     *
     */
    template<typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type
    log_falling_factorial(const T1 x, const T2 n) { 
      using boost::math::lgamma;
      return lgamma(x + 1) - lgamma(n + 1); 
    }

  }
}

#endif
