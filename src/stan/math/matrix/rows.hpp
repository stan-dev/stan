#ifndef __STAN__MATH__MATRIX__ROWS_HPP__
#define __STAN__MATH__MATRIX__ROWS_HPP__

#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    template <typename T, int R, int C>
    inline 
    size_t 
    rows(const Eigen::Matrix<T,R,C>& m) {
      return m.rows();
    }
    
  }
}
#endif
