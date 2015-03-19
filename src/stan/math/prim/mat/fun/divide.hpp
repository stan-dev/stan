#ifndef STAN__MATH__PRIM__MAT__FUN__DIVIDE_HPP
#define STAN__MATH__PRIM__MAT__FUN__DIVIDE_HPP

#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/utility/enable_if.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace math {

    /**
     * Return specified matrix divided by specified scalar.
     * @tparam R Row type for matrix.
     * @tparam C Column type for matrix.
     * @param m Matrix.
     * @param c Scalar.
     * @return Matrix divided by scalar.
     */
    template <int R, int C, typename T>
    inline
    typename boost::enable_if_c<boost::is_arithmetic<T>::value,
                                Eigen::Matrix<double, R, C> >::type
    divide(const Eigen::Matrix<double,R,C>& m,
           T c) {
      return m / c;
    }

  }
}
#endif
