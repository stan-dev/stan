#ifndef __STAN__MATH__MATRIX__VALIDATE_VECTOR_HPP__
#define __STAN__MATH__MATRIX__VALIDATE_VECTOR_HPP__

#include <sstream>
#include <stdexcept>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    template <typename T, int R, int C>
    inline void validate_vector(const Eigen::Matrix<T,R,C>& x,
                                const char* msg) {
      if (x.rows() == 1 || x.cols() == 1) return;
      std::stringstream ss;
      ss << "error in " << msg
         << "; require vector, found "
         << " rows=" << x.rows() << "cols=" << x.cols();
      throw std::domain_error(ss.str());
    }    
    
  }
}
#endif
