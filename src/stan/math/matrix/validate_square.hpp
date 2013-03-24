#ifndef __STAN__MATH__MATRIX__VALIDATE_SQUARE_HPP__
#define __STAN__MATH__MATRIX__VALIDATE_SQUARE_HPP__

#include <sstream>
#include <stdexcept>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {
    
    template <typename T, int R, int C>
    void validate_square(const Eigen::Matrix<T,R,C>& x,
                         const char* msg) {
      if (x.rows() == x.cols()) return;
      std::stringstream ss;
      ss << "error in call to " << msg
         << "; require square matrix, but found"
         << " rows=" << x.rows()
         << "; cols=" << x.cols();
      throw std::domain_error(ss.str());
    }
    
  }
}
#endif
