#ifndef STAN__ERROR_HANDLING__SCALAR__CHECK_CONSISTENT_SIZE_HPP
#define STAN__ERROR_HANDLING__SCALAR__CHECK_CONSISTENT_SIZE_HPP

#include <sstream>
#include <stan/error_handling/invalid_argument.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the dimension of x is consistent, which
     * is defined to be <code>expected_size</code> if x is a vector or 1 if x
     * is not a vector.
     *
     * @tparam T Type of value
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param x Variable to check for consistent size
     * @param expected_size Expected size if x is a vector
     *
     * @return <code>true</code> if x is scalar or if x is vector-like and
     *   has size of <code>expected_size</code>
     * @throw <code>invalid_argument</code> if the size is inconsistent
     */
    template <typename T>
    inline bool check_consistent_size(const std::string& function,
                                      const std::string& name,
                                      const T& x,
                                      size_t expected_size) {
      if (!is_vector<T>::value)
        return true;
      if (is_vector<T>::value && expected_size == stan::size_of(x))
        return true;
      
      std::stringstream msg;
      msg << ", expecting dimension = "
          << expected_size
          << "; a function was called with arguments of different "
          << "scalar, array, vector, or matrix types, and they were not "
          << "consistently sized;  all arguments must be scalars or "
          << "multidimensional values of the same shape.";

      invalid_argument(function, name, stan::size_of(x),
                       "has dimension = ",
                       msg.str());
      
      return false;
    }

  }
}
#endif
