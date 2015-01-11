#ifndef STAN__MATH__FUNCTIONS__LGAMMA_HPP
#define STAN__MATH__FUNCTIONS__LGAMMA_HPP

#include <boost/math/special_functions/gamma.hpp>

namespace stan {

  namespace math {

    /**
     *
       \f[
       \mbox{lgamma}(x) = 
       \begin{cases}
         \textrm{error} & \mbox{if } x\in \{\dots,-3,-2,-1,0\}\\
         \ln\Gamma(x) & \mbox{if } x\not\in \{\dots,-3,-2,-1,0\}\\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{lgamma}(x)}{\partial x} = 
       \begin{cases}
         \textrm{error} & \mbox{if } x\in \{\dots,-3,-2,-1,0\}\\
         \Psi(x) & \mbox{if } x\not\in \{\dots,-3,-2,-1,0\}\\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
    \f]
     */
    // throws domain_error if x is at pole
    double lgamma(double x) {
      return boost::math::lgamma(x);
    }

  }
}

#endif
