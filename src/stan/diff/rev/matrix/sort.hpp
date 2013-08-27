#ifndef __STAN__DIFF__REV__SORT_HPP__
#define __STAN__DIFF__REV__SORT_HPP__

#include <valarray>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>
#include <stan/diff/rev/operator_greater_than.hpp>
#include <stan/diff/rev/operator_less_than.hpp>
#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <algorithm>    // std::sort
#include <functional>   // std::greater

namespace stan {
  namespace diff {
   
    /**
     * Return the specified standard vector in ascending order with gradients kept.
     *
     * @param xs Standard vector to order.
     * @return Standard vector ordered.
     * @tparam T Type of elements of the vector.
     */
    inline std::vector<var> sort_asc(std::vector<var> xs) {
      std::sort(xs.begin(), xs.end());
      return xs;
    }
    
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
     * Return the specified eigen vector in ascending order with gradients kept.
     *
     * @param xs Eigen vector to order.
     * @return Eigen vector ordered.
     * @tparam T Type of elements of the vector.
     */
    template <int R, int C>
    inline typename Eigen::Matrix<var,R,C> sort_asc(Eigen::Matrix<var,R,C> xs) {
      std::sort(xs.data(), xs.data()+xs.size());
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
