#ifndef STAN__MATH__META__VALUE_TYPE_HPP
#define STAN__MATH__META__VALUE_TYPE_HPP

#include <vector>

namespace stan {

  namespace math {

    /**
     * Primary template class for metaprogram to compute the type of
     * values stored in a container.
     *
     * Only the specializations have behavior that can be used, and
     * all implement a typedef <code>type</code> for the type of the
     * values in the container.
     *
     * tparam T type of container.
     */
    template <typename T>
    struct value_type {
    };


    /**
     * Template class for metaprogram to compute the type of values
     * stored in a constant container.
     *
     * @tparam T type of container without const modifier.
     */
    template <typename T>
    struct value_type<const T> {
      typedef typename value_type<T>::type type;
    };


    /**
     * Template metaprogram class to compute the type of values stored
     * in a standard vector.
     *
     * @tparam T type of elements in standard vector.
     */
    template <typename T>
    struct value_type<std::vector<T> > {

      /**
       * Type of value stored in a standard vector with type
       * <code>T</code> entries. 
       */
      typedef typename std::vector<T>::value_type type;

    };

    
  }
}


#endif
