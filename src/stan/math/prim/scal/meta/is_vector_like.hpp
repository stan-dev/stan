#ifndef STAN__MATH__PRIM__SCAL__META__IS_VECTOR_LIKE_HPP
#define STAN__MATH__PRIM__SCAL__META__IS_VECTOR_LIKE_HPP

#include <stan/math/prim/scal/meta/is_vector.hpp>

namespace stan {

  // ****************** additions for new VV *************************

  // handles scalar, eigen vec, eigen row vec, std vec
  template <typename T>
  struct is_vector_like {
    enum { value = stan::is_vector<T>::value };  
  };
  template <typename T>
  struct is_vector_like<T*> {
    enum { value = true };
  };
  // handles const
  template <typename T>
  struct is_vector_like<const T> {
    enum { value = stan::is_vector_like<T>::value };  
  };
}
#endif

