#ifndef __STAN__MATH__MATRIX__DIVIDE_HPP__
#define __STAN__MATH__MATRIX__DIVIDE_HPP__

#include <boost/type_traits/is_arithmetic.hpp> 
#include <boost/utility/enable_if.hpp>
#include <stan/math/matrix/Eigen.hpp>

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
