#ifndef STAN__MATH__FUNCTIONS__FMA_HPP
#define STAN__MATH__FUNCTIONS__FMA_HPP

#include <cmath>

namespace stan {
  namespace math {

    /**
     * The fused multiply-add operation (C99).   
     *
     * This double-based operation delegates to <code>std::fma</code>.
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
    double
    fma(double a, double b, double c) {
      return std::fma(a,b,c);
    }

  }
}
#endif
