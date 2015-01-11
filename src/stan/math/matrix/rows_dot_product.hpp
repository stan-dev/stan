#ifndef STAN__MATH__MATRIX__ROWS_DOT_PRODUCT_HPP
#define STAN__MATH__MATRIX__ROWS_DOT_PRODUCT_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/error_handling/matrix/check_matching_sizes.hpp>

namespace stan {
  namespace math {
    
    /**
     * Returns the dot product of the specified vectors.
     *
     * @param v1 First vector.
     * @param v2 Second vector.
     * @return Dot product of the vectors.
     * @throw std::domain_error If the vectors are not the same
     * size or if they are both not vector dimensioned.
     */
    template<int R1,int C1,int R2, int C2>
    inline Eigen::Matrix<double, R1, 1>
    rows_dot_product(const Eigen::Matrix<double, R1, C1>& v1, 
                     const Eigen::Matrix<double, R2, C2>& v2) {
      stan::error_handling::check_matching_sizes("rows_dot_product",
                                                 "v1", v1,
                                                 "v2", v2);
      Eigen::Matrix<double, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        ret(j) = v1.row(j).dot(v2.row(j));
      }
      return ret;
    }    
    
  }
}
#endif
