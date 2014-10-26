#ifndef STAN__ERROR_HANDLING__CHECK_CONSISTENT_SIZE_HPP
#define STAN__ERROR_HANDLING__CHECK_CONSISTENT_SIZE_HPP

#include <sstream>
#include <stan/error_handling/dom_err.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace math {

    // NOTE: this will not throw if nan is passed in.
    template <typename T, typename T_result>
    inline bool check_consistent_size(size_t max_size,
                                      const char* function,
                                      const T& x,
                                      const char* name,
                                      T_result* result) {
      size_t x_size = stan::size_of(x);
      if (is_vector<T>::value && x_size == max_size)
        return true;
      if (!is_vector<T>::value && x_size == 1)
        return true;
      
      std::stringstream msg;
      msg << " dimension=%1%, expecting dimension of either 1 or "
          << "max_size=" << max_size
          << "; a vectorized function was called with arguments of different "
          << "scalar, array, vector, or matrix types, and they were not "
          << "consistently sized;  all arguments must be scalars or "
          << "multidimensional values of the same shape.";

      std::string tmp(msg.str());

      return dom_err(function,x_size,name,
                     tmp.c_str(), "",                     
                     result);
    }

  }
}
#endif
