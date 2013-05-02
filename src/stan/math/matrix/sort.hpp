#ifndef __STAN__MATH__MATRIX__SORT_HPP__
#define __STAN__MATH__MATRIX__SORT_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <algorithm>    // std::sort
#include <functional>   // std::greater

namespace stan {
  namespace math {
   
    /**
     * Return the specified standard vector in ascending order.
     *
     * @param xs Standard vector to order.
     * @return Standard vector ordered.
     * @tparam T Type of elements of the vector.
     */
    template <typename T>
    inline T sort_asc(std::vector<T> xs)
    {
	std::sort (xs.begin(), xs.end());
      return xs;
    }

    /**
     * Return the specified standard vector in descending order.
     *
     * @param xs Standard vector to order.
     * @return Standard vector ordered.
     * @tparam T Type of elements of the vector.
     */
    template <typename T>
    inline T sort_desc(std::vector<T> xs)
    {
	std::sort (xs.begin(), xs.end(), std::greater<T>());
      return xs;
    }

    /**
     * Return the specified eigen vector in ascending order.
     *
     * @param xs Eigen vector to order.
     * @return Eigen vector ordered.
     * @tparam T Type of elements of the vector.
     */
 template <typename T, int R, int C>
    inline T sort_asc(Eigen::Matrix<T,R,C> xs)
    {
	std::sort (xs.data(), xs.data()+xs.size());
      return xs;
    }

    /**
     * Return the specified eigen vector in descending order.
     *
     * @param xs Eigen vector to order.
     * @return Eigen vector ordered.
     * @tparam T Type of elements of the vector.
     */
 template <typename T, int R, int C>
    inline T sort_desc(Eigen::Matrix<T,R,C> xs)
    {
	std::sort (xs.data(), xs.data()+xs.size(), std::greater<T>());
      return xs;
    }
    
  }
}
#endif
