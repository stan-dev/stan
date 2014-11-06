#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_MATCHING_DIMS_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_MATCHING_DIMS_HPP

#include <sstream>
#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/matrix/check_size_match.hpp>

namespace stan {
  namespace error_handling {

    // NOTE: this will not throw  if y1 or y2 contains nan values.
    template <typename T1, typename T2, int R1, int C1, int R2, int C2>
    inline bool check_matching_dims(const std::string& function,
                                    const std::string& name1,
                                    const Eigen::Matrix<T1,R1,C1>& y1,
                                    const std::string& name2,
                                    const Eigen::Matrix<T2,R2,C2>& y2) {
      check_size_match(function, 
                       "Rows of matrix 1", y1.rows(),
                       "Rows of matrix 2", y2.rows());
      check_size_match(function, 
                       "Columns of matrix 1", y1.cols(),
                       "Columns of matrix 2", y2.cols());
      return true;
    }

  }
}
#endif
