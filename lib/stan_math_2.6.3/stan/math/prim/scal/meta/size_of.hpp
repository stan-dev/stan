#ifndef STAN_MATH_PRIM_SCAL_META_SIZE_OF_HPP
#define STAN_MATH_PRIM_SCAL_META_SIZE_OF_HPP

#include <stan/math/prim/scal/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>

namespace stan {



  template<typename T, bool is_vec>
  struct size_of_helper {
    static size_t size_of(const T& /*x*/) {
      return 1U;
    }
  };

  template<typename T>
  struct size_of_helper<T, true> {
    static size_t size_of(const T& x) {
      return x.size();
    }
  };

  template <typename T>
  size_t size_of(const T& x) {
    return size_of_helper<T, is_vector<T>::value>::size_of(x);
  }

}
#endif

