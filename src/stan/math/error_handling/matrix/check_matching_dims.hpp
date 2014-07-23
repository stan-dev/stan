#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_MATCHING_DIMS_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_MATCHING_DIMS_HPP

#include <sstream>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/matrix/check_size_match.hpp>

namespace stan {
  namespace math {

    template <typename T1, typename T2, int R1, int C1, int R2, int C2,
              typename T_result>
    inline bool check_matching_dims(const char* function,
                                    const Eigen::Matrix<T1,R1,C1>& y1,
                                    const char* name1,
                                    const Eigen::Matrix<T2,R2,C2>& y2,
                                    const char* name2,
                                    T_result* result) {
      stan::math::check_size_match(function, 
                       y1.rows(), "Rows of matrix 1",
                       y2.rows(), "Rows of matrix 2",
                       result);
      stan::math::check_size_match(function, 
                       y1.cols(), "Columns of matrix 1",
                       y2.cols(), "Columns of matrix 2",
                       result);
      return true;
    }

  }
}
#endif
