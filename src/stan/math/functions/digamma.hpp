#ifndef STAN__MATH__FUNCTIONS__DIGAMMA_HPP
#define STAN__MATH__FUNCTIONS__DIGAMMA_HPP

#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace math {
    
    /**
     *
       \f[
       \mbox{digamma}(x) = 
       \begin{cases}
         \textrm{error} & \mbox{if } x\in \{\dots,-3,-2,-1,0\}\\
         \Psi(x) & \mbox{if } x\not\in \{\dots,-3,-2,-1,0\}\\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{digamma}(x)}{\partial x} = 
       \begin{cases}
         \textrm{error} & \mbox{if } x\in \{\dots,-3,-2,-1,0\}\\
         \frac{\partial\, \Psi(x)}{\partial x} & \mbox{if } x\not\in \{\dots,-3,-2,-1,0\}\\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
   
       \f[
       \Psi(x)=\frac{\Gamma'(x)}{\Gamma(x)}
       \f]
       
       \f[
       \frac{\partial \, \Psi(x)}{\partial x} = \frac{\Gamma''(x)\Gamma(x)-(\Gamma'(x))^2}{\Gamma^2(x)}
       \f]
    */
    double digamma(double x) {
      return boost::math::digamma(x);
    }

  }
}

#endif
