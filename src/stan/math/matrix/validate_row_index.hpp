#ifndef __STAN__MATH__MATRIX__VALIDATE_ROW_INDEX_HPP__
#define __STAN__MATH__MATRIX__VALIDATE_ROW_INDEX_HPP__

#include <sstream>
#include <stdexcept>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace math {
    
    template <typename T, int R, int C>
    void validate_row_index(const Eigen::Matrix<T,R,C>& m,
                            size_t i,
                            const char* msg) {
      if (i > 0 && i <=  static_cast<size_t>(m.rows())) return;
      std::stringstream ss;
      ss << "require 0 < row index <= number of rows in " << msg;
      ss << " found rows()=" << m.rows()
         << "; index i=" << i;
      throw std::domain_error(ss.str());
    }
    
  }
}
#endif
