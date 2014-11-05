#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_VECTOR_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_VECTOR_HPP

#include <sstream>
#include <stan/meta/traits.hpp>
#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace error_handling {

    // NOTE: this will not throw if x contains nan values.
    template <typename T, int R, int C>
    inline bool check_vector(const std::string& function,
                             const std::string& name,
                             const Eigen::Matrix<T,R,C>& x) {
      if (x.rows() == 1 || x.cols() == 1)
        return true;
      
      std::ostringstream msg;
      msg << ") has " << x.rows() << " rows and " 
          << x.cols() << " columns but it should be a vector so it should "
          << "either have 1 row or 1 column";
      dom_err(function,
              name,
              typename scalar_type<T>::type(),
              "(", msg.str());
      return false;
    }

  }
}
#endif
