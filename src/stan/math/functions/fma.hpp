#ifndef STAN__MATH__FUNCTIONS__FMA_HPP
#define STAN__MATH__FUNCTIONS__FMA_HPP

#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {

    /**
     * The fused multiply-add operation (C99).  
     *
     * The function is defined by
     *
     * <code>fma(a,b,c) = (a * b) + c</code>.
     *
     *
       \f[
       \mbox{fma}(x,y,z) = 
       \begin{cases}
         x\cdot y+z & \mbox{if } -\infty\leq x,y,z \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{fma}(x,y,z)}{\partial x} = 
       \begin{cases}
         y & \mbox{if } -\infty\leq x,y,z \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{fma}(x,y,z)}{\partial y} = 
       \begin{cases}
         x & \mbox{if } -\infty\leq x,y,z \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{fma}(x,y,z)}{\partial z} = 
       \begin{cases}
         1 & \mbox{if } -\infty\leq x,y,z \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a First value.
     * @param b Second value.
     * @param c Third value.
     * @return Product of the first two values plust the third.
     */
    template <typename T1, typename T2, typename T3>
    inline typename boost::math::tools::promote_args<T1,T2,T3>::type
    fma(const T1 a, const T2 b, const T3 c) {
      return (a * b) + c;
    }

  }
}
#endif
