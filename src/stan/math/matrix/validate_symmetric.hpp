#ifndef __STAN__MATH__MATRIX__VALIDATE_SYMMETRIC_HPP__
#define __STAN__MATH__MATRIX__VALIDATE_SYMMETRIC_HPP__

#include <sstream>
#include <stdexcept>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace math {
    
    template <typename T, int R, int C>
    void validate_symmetric(const Eigen::Matrix<T,R,C>& x,
                            const char* msg) {
      // tolerance = 1E-8
      validate_square(x,msg);
      for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
          if (x(i,j) != x(j,i)) {
            std::stringstream ss;
            ss << "error in call to " << msg
               << "; require symmetric matrix, but found"
               << "; x[" << i << "," << j << "]=" << x(i,j)
               << "; x[" << j << "," << i << "]=" << x(j,i);
            throw std::domain_error(ss.str());
          }
        }
      }    
    }
    
  }
}
#endif
