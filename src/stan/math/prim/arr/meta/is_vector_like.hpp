#ifndef STAN__MATH__PRIM__ARR__META__IS_VECTOR_LIKE_HPP
#define STAN__MATH__PRIM__ARR__META__IS_VECTOR_LIKE_HPP

namespace stan {

  template <typename T>
  struct is_vector_like<std::vector<T> > {
    enum { value = true };
  };

}
#endif

