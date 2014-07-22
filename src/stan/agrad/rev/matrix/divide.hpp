#ifndef STAN__AGRAD__REV__MATRIX__DIVIDE_HPP
#define STAN__AGRAD__REV__MATRIX__DIVIDE_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/operators/operator_division.hpp>
#include <stan/agrad/rev/matrix/to_var.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

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
