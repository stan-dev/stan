#ifndef STAN__MATH__PRIM__MAT__FUN__SORT_INDICES_ASC_HPP
#define STAN__MATH__PRIM__MAT__FUN__SORT_INDICES_ASC_HPP


#include <vector>
#include <algorithm>    // std::sort
#include <iostream>

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <stan/math/prim/mat/fun/sort_indices.hpp>

namespace stan {

  namespace math {
    
    /**
     * Return a sorted copy of the argument container in ascending order.
     *
     * @tparam C type of container
     * @param xs Container to sort
     * @return sorted version of container
     */
    template <typename C>
    std::vector<int> sort_indices_asc(const C& xs) {
      return sort_indices<true>(xs);
    }

  }
}
#endif
