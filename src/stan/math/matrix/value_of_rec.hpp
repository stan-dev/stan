#ifndef STAN__MATH__MATRIX__VALUE_OF_REC_HPP
#define STAN__MATH__MATRIX__VALUE_OF_REC_HPP

#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {
    
    /**
     * Convert a matrix to a matrix of doubles.
     * Math double version is the identity function 
     *
     * @tparam R Rows of matrix
     * @tparam C Columns of matrix
     * @param[in] M Matrix to be converted
     **/
    template<int R,int C>
    inline const Eigen::Matrix<double,R,C> &value_of_rec(const Eigen::Matrix<double,R,C> &M) {
      return M;
    }
  }
}

#endif
