#ifndef STAN__MATH__PRIM__SCAL__META__IS_VECTOR_HPP
#define STAN__MATH__PRIM__SCAL__META__IS_VECTOR_HPP

namespace stan {

  // FIXME: use boost::type_traits::remove_all_extents to
  //        extend to array/ptr types

  template <typename T>
  struct is_vector {
    enum { value = 0 };
    typedef T type;
  };
}
#endif

