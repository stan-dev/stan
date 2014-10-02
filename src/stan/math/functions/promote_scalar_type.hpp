#ifndef STAN__MATH__FUNCTIONS__PROMOTE_SCALAR_TYPE_HPP
#define STAN__MATH__FUNCTIONS__PROMOTE_SCALAR_TYPE_HPP

#include <vector>

namespace stan {
  
  namespace math {


    /**
     * Template metaprogram to calculate a type for converting a
     * convertible type.  This is the base case.
     *
     * @tparam T result scalar type.
     * @tparam S input type
     */
    template <typename T, typename S>
    struct promote_scalar_type {

      /**
       * The promoted type.
       */
      typedef T type;

    };


    /**
     * Template metaprogram to calculate a type for a container whose
     * underlying scalar is converted from the second template
     * parameter type to the first. 
     *
     * @tparam T result scalar type.
     * @tparam S input type
     */
    template <typename T, typename S>
    struct promote_scalar_type<T, std::vector<S> > {

      /**
       * The promoted type.
       */
      typedef std::vector<typename promote_scalar_type<T,S>::type> type;

    };


  }

}

#endif
