#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_MULTIPLICABLE_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_MULTIPLICABLE_HPP

#include <sstream>
#include <stan/meta/traits.hpp>
#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/matrix/check_size_match.hpp>

namespace stan {
  namespace error_handling {

    // NOTE: this will not throw if y1 or y2 contains nan values.
    template <typename T1, typename T2>
    inline bool check_multiplicable(const std::string& function,
                                    const std::string& name1,
                                    const T1& y1,
                                    const std::string& name2,
                                    const T2& y2) {
      check_size_match(function, 
                       "Columns of matrix 1", y1.cols(), 
                       "Rows of matrix 2", y2.rows());
      return true;
    }
  }
}
#endif
