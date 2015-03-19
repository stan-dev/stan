#ifndef STAN__MATH__REV__MAT__FUN__DIVIDE_HPP
#define STAN__MATH__REV__MAT__FUN__DIVIDE_HPP

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/fun/to_var.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return the division of the first scalar by
     * the second scalar.
     * @param[in] x Specified vector.
     * @param[in] y Specified scalar.
     * @return Vector divided by the scalar.
     */
    inline double
    divide(double x, double y) {
      return x / y;
    }
    template <typename T1, typename T2>
    inline var
    divide(const T1& v, const T2& c) {
      return to_var(v) / to_var(c);
    }
    /**
     * Return the division of the specified column vector by
     * the specified scalar.
     * @param[in] v Specified vector.
     * @param[in] c Specified scalar.
     * @return Vector divided by the scalar.
     */
    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<var,R,C>
    divide(const Eigen::Matrix<T1, R,C>& v, const T2& c) {
      return to_var(v) / to_var(c);
    }

  }
}
#endif
