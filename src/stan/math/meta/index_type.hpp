#ifndef STAN__MATH__META__INDEX_TYPE_HPP
#define STAN__MATH__META__INDEX_TYPE_HPP

#include <vector>

namespace stan {

  namespace math {

    /**
     * Primary template class for the metaprogram to compute the index
     * type of a container.
     *
     * Only the specializations have behavior that can be used, and
     * all implement a typedef <code>type</code> for the type of the
     * index given container <code>T</code>.
     *
     * tparam T type of container.
     */
    template <typename T>
    struct index_type {
    };


    /**
     * Template class for metaprogram to compute the type of indexes
     * used in a constant container type. 
     *
     * @tparam T type of container without const modifier.
     */
    template <typename T>
    struct index_type<const T> {
      typedef typename index_type<T>::type type;
    };




    /**
     * Template metaprogram class to compute the type of index for a
     * standard vector.
     *
     * @tparam T type of elements in standard vector.
     */
    template <typename T>
    struct index_type<std::vector<T> > {

      /**
       * Typedef for index of standard vectors.
       */
      typedef typename std::vector<T>::size_type type;

    };

    
  }
}


#endif
