#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_VECTOR_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_VECTOR_HPP

#include <sstream>
#include <stan/meta/traits.hpp>
#include <stan/error_handling/invalid_argument.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the matrix is either a
     * row vector or column vector.
     *
     * This function checks the runtime size of the matrix to check
     * whether it is a row or column vector.
     * 
     * @tparam T Scalar type of the matrix
     * @tparam R Compile time rows of the matrix
     * @tparam C Compile time columns of the matrix
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param x Matrix
     *
     * @return <code>true</code> if x either has 1 columns or 1 rows
     * @throw <code>std::invalid_argument</code> if x is not a row or column
     *   vector.
     */
    template <typename T, int R, int C>
    inline bool check_vector(const char* function,
                             const char* name,
                             const Eigen::Matrix<T,R,C>& x) {
      if (R == 1)
        return true;
      if (C == 1)
        return true;
      if (x.rows() == 1 || x.cols() == 1)
        return true;
      
      std::ostringstream msg;
      msg << ") has " << x.rows() << " rows and " 
          << x.cols() << " columns but it should be a vector so it should "
          << "either have 1 row or 1 column";
      std::string msg_str(msg.str());
      invalid_argument(function,
                       name,
                       typename scalar_type<T>::type(),
                       "(", msg_str.c_str());
      return false;
    }

  }
}
#endif
