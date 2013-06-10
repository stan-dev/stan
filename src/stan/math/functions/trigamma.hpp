#ifndef __STAN__MATH__FUNCTIONS__TRIGAMMA_HPP__
#define __STAN__MATH__FUNCTIONS__TRIGAMMA_HPP__

#include <stan/math/constants.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline
    T
    trigamma(T x) {

      using std::floor;
      using std::sin;

      double small = 0.0001;
      double large = 5.0;
      double b2 =  1.0 / 6.0;
      double b4 = -1.0 / 30.0;
      double b6 =  1.0 / 42.0;
      double b8 = -1.0 / 30.0;
      T value;
      T y;
      T z;

      //negative integers and zero
      if ((x <= 0.0) && (floor(x) == x))
        return positive_infinity();

      //negative non-integers
      if((x <= 0) && (floor(x) != x))
        return -trigamma(-x + 1.0) + (pi() / sin(-pi() * x)) 
          * (pi() / sin(-pi() * x));

      z = x;

      //small value approximation if x <= small.
      if (x <= small)
          return 1.0 / x / x;

      //increase argument to (x+1) >= large.
      value = 0.0;
  
      while (z < large) {
          value += 1.0 / z / z;
          z += 1.0;
        }

      //asymptotic formula if x >= large
      y = 1.0 / z / z;
      value += 0.5 * y + (1.0 + y * (b2  + y * (b4 + y * (b6 + y * b8)))) / z;
      
      return value;
    }
  }
}

#endif
