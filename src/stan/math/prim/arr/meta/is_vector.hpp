#ifndef STAN__MATH__PRIM__ARR__META__IS_VECTOR_HPP
#define STAN__MATH__PRIM__ARR__META__IS_VECTOR_HPP

#include <stan/math/prim/scal/meta/is_vector.hpp>
#include <vector>

namespace stan {

  // FIXME: use boost::type_traits::remove_all_extents to
  //   extend to array/ptr types

  template <typename T>
  struct is_vector<const T> {
    enum { value = is_vector<T>::value };
    typedef T type;
  };
  template <typename T>
  struct is_vector<std::vector<T> > {
    enum { value = 1 };
    typedef T type;
  };
}
#endif

