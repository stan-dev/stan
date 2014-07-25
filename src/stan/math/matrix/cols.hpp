#ifndef STAN__MATH__MATRIX__COLS_HPP
#define STAN__MATH__MATRIX__COLS_HPP

#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {
    
    template <typename T, int R, int C>
    inline 
    size_t 
    cols(const Eigen::Matrix<T,R,C>& m) {
      return m.cols();
    }
    
  }
}
#endif
