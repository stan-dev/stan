#ifndef __STAN__MATH__MATRIX__VALIDATE_MULTIPLICABLE_HPP__
#define __STAN__MATH__MATRIX__VALIDATE_MULTIPLICABLE_HPP__

#include <sstream>
#include <stdexcept>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {
    
    template <typename T1, int R1, int C1, typename T2, int R2, int C2>
    inline void validate_multiplicable(const Eigen::Matrix<T1,R1,C1>& x1,
                                       const Eigen::Matrix<T2,R2,C2>& x2,
                                       const char* msg) {
      if (x1.cols() == x2.rows()) return;
      std::stringstream ss;
      ss << "error in call to " << msg
         << "; require cols of arg1 to match rows of arg2, but found "
         << " arg1 rows=" << x1.rows() << " arg1 cols=" << x1.cols()
         << " arg2 rows=" << x2.rows() << " arg2 cols=" << x2.cols();
      throw std::domain_error(ss.str());
    }    
    
  }
}
#endif
