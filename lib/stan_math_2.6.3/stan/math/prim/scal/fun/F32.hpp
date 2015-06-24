#ifndef STAN_MATH_PRIM_SCAL_FUN_F32_HPP
#define STAN_MATH_PRIM_SCAL_FUN_F32_HPP

#include <cmath>

namespace stan {

  namespace math {

    template<typename T>
    T F32(T a, T b, T c, T d, T e, T z, T precision = 1e-6) {
      using std::exp;
      using std::log;
      using std::fabs;

      T F = 1.0;

      T tNew = 0.0;

      T logT = 0.0;

      T logZ = log(z);

      int k = 0.0;

      while (fabs(tNew) > precision || k == 0) {
        T p = (a + k) * (b + k) * (c + k) / ( (d + k) * (e + k) * (k + 1) );

        // If a, b, or c is a negative integer then the series terminates
        // after a finite number of interations
        if (p == 0) break;

        logT +=  (p > 0 ? 1.0 : -1.0) * log(fabs(p)) + logZ;

        tNew = exp(logT);

        F += tNew;

        ++k;
      }
      return F;
    }

  }
}
#endif
