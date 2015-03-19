#ifndef STAN__MATH__PRIM__SCAL__FUN__TRIGAMMA_HPP
#define STAN__MATH__PRIM__SCAL__FUN__TRIGAMMA_HPP

  // Reference:
  //   BE Schneider,
  //   Algorithm AS 121:
  //   Trigamma Function,
  //   Applied Statistics,
  //   Volume 27, Number 1, pages 97-99, 1978.

#include <stan/math/prim/scal/fun/constants.hpp>

namespace stan {

  namespace math {

    /**
     *
     *
       \f[
       \mbox{trigamma}(x) =
       \begin{cases}
         \textrm{error} & \mbox{if } x\in \{\dots,-3,-2,-1,0\}\\
         \Psi_1(x) & \mbox{if } x\not\in \{\dots,-3,-2,-1,0\}\\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{trigamma}(x)}{\partial x} =
       \begin{cases}
         \textrm{error} & \mbox{if } x\in \{\dots,-3,-2,-1,0\}\\
         \frac{\partial\, \Psi_1(x)}{\partial x} & \mbox{if } x\not\in \{\dots,-3,-2,-1,0\}\\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \Psi_1(x)=\sum_{n=0}^\infty \frac{1}{(x+n)^2}
       \f]

       \f[
       \frac{\partial \, \Psi_1(x)}{\partial x} = -2\sum_{n=0}^\infty \frac{1}{(x+n)^3}
       \f]
     */
    template <typename T>
    inline
    T
    trigamma(T x) {

      using std::floor;
      using std::sin;

      double small = 0.0001;
      double large = 5.0;
      T value;
      T y;
      T z;

      //bernoulli numbers
      double b2 =  1.0 / 6.0;
      double b4 = -1.0 / 30.0;
      double b6 =  1.0 / 42.0;
      double b8 = -1.0 / 30.0;

      //negative integers and zero return postiive infinity
      //see http://mathworld.wolfram.com/PolygammaFunction.html
      if ((x <= 0.0) && (floor(x) == x)) {
        value = positive_infinity();
        return value;
      }

      //negative non-integers: use the reflection formula
      //see http://mathworld.wolfram.com/PolygammaFunction.html
      if((x <= 0) && (floor(x) != x)) {
        value = -trigamma(-x + 1.0) + (pi() / sin(-pi() * x))
          * (pi() / sin(-pi() * x));
        return value;
      }

      //small value approximation if x <= small.
      if (x <= small)
        return 1.0 / (x * x);

      //use recurrence relation until x >= large
      //see http://mathworld.wolfram.com/PolygammaFunction.html
      z = x;
      value = 0.0;
      while (z < large) {
        value += 1.0 / (z * z);
        z += 1.0;
      }

      //asymptotic expansion as a Laurent series if x >= large
      //see http://en.wikipedia.org/wiki/Trigamma_function
      y = 1.0 / (z * z);
      value += 0.5 * y + (1.0 + y * (b2  + y * (b4 + y * (b6 + y * b8)))) / z;

      return value;
    }
  }
}

#endif
