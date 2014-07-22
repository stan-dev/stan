#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_VECTOR_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_VECTOR_HPP

#include <sstream>
#include <stan/meta/traits.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    template <typename T, int R, int C, typename T_result>
    inline bool check_vector(const char* function,
                             const Eigen::Matrix<T,R,C>& x,
                             const char* name,
                             T_result* result) {
      if (x.rows() == 1 || x.cols() == 1)
        return true;

      std::ostringstream msg;
      msg << name << " (%1%) has " << x.rows() << " rows and " 
          << x.cols() << " columns but it should be a vector so it should either have 1 row or 1 column";
      std::string tmp(msg.str());
      return dom_err(function, 
                     typename scalar_type<T>::type(),
                     name,
                     tmp.c_str(),"",
                     result);
    }

  }
}
#endif
