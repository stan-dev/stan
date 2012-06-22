#ifndef __STAN__META__TRAITS_HPP__
#define __STAN__META__TRAITS_HPP__

#include <boost/type_traits.hpp>

namespace stan {

  /**
   * Metaprogramming struct to detect whether a given type is constant
   * in the mathematical sense (not the C++ <code>const</code>
   * sense). If the parameter type is constant, <code>value</code>
   * will be equal to <code>true</code>.
   *
   * The baseline implementation in this abstract base class is to
   * classify a type <code>T</code> as constant if it can be converted
   * (i.e., assigned) to a <code>double</code>.  This baseline should
   * be overridden for any type that should be treated as a variable.
   *
   * @tparam T Type being tested.
   */
  template <typename T>
  struct is_constant {
    /**
     * A boolean constant with equal to <code>true</code> if the
     * type parameter <code>T</code> is a mathematical constant.
     */
    enum { value = boost::is_convertible<T,double>::value };
  };

  // FIXME: use boost::type_traits::remove_all_extents to extend to array/ptr types

  /**
   * Metaprogram structure to determine the base scalar type
   * of a template argument.
   *
   * <p>This base class should be specialized for structured types.
   *
   * @tparam T Type of object.
   */
  template <typename T>
  struct scalar_type {
    /** 
     * Base scalar type for object.
     */
    typedef T type;
  };

  /**
   * Metaprogram specialization extracting the base type of
   * a standard vector recursively.
   *
   * @tparam Scalar type of vector.
   */
  template <typename T>
  struct scalar_type<std::vector<T> > {
    typedef typename scalar_type<T>::type type;
  };
      


}

#endif
