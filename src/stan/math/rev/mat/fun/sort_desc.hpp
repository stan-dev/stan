#ifndef STAN__MATH__REV__MAT__FUN__SORT_DESC_HPP
#define STAN__MATH__REV__MAT__FUN__SORT_DESC_HPP

#include <valarray>
#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/core/v_vari.hpp>
#include <stan/math/rev/core/operator_greater_than.hpp>
#include <stan/math/rev/core/operator_less_than.hpp>
#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <algorithm>    // std::sort
#include <functional>   // std::greater

namespace stan {
  namespace agrad {
   
    /**
     * Return the specified standard vector in descending order with gradients kept.
     *
     * @param xs Standard vector to order.
     * @return Standard vector ordered.
     * @tparam T Type of elements of the vector.
     */
    inline std::vector<var> sort_desc(std::vector<var> xs) {
      std::sort(xs.begin(), xs.end(), std::greater<var>());
      return xs;
    }

    /**
     * Return the specified eigen vector in descending order with gradients kept.
     *
     * @param xs Eigen vector to order.
     * @return Eigen vector ordered.
     * @tparam T Type of elements of the vector.
     */
    template <int R, int C>
    inline typename Eigen::Matrix<var,R,C> sort_desc(Eigen::Matrix<var,R,C> xs) {
      std::sort(xs.data(), xs.data()+xs.size(), std::greater<var>());
      return xs;
    }
    
  }
}
#endif
