#ifndef STAN__ERROR_HANDLING__SCALAR__CHECK_CONSISTENT_SIZE_HPP
#define STAN__ERROR_HANDLING__SCALAR__CHECK_CONSISTENT_SIZE_HPP

#include <sstream>
#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace error_handling {

    // NOTE: this will not throw if nan is passed in.
    template <typename T>
    inline bool check_consistent_size(const std::string& function,
                                      const std::string& name,
                                      const T& x,
                                      size_t max_size) {
      size_t x_size = stan::size_of(x);
      if (is_vector<T>::value && x_size == max_size)
        return true;
      if (!is_vector<T>::value && x_size == 1)
        return true;
      
      std::stringstream msg;
      msg << ", expecting dimension of either 1 or "
          << "max_size=" << max_size
          << "; a vectorized function was called with arguments of different "
          << "scalar, array, vector, or matrix types, and they were not "
          << "consistently sized;  all arguments must be scalars or "
          << "multidimensional values of the same shape.";

      dom_err(function, name, x_size,
              "dimension=",
              msg.str());
      
      return false;
    }

  }
}
#endif
