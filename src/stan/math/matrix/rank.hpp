#ifndef STAN__MATH__MATRIX__RANK_HPP
#define STAN__MATH__MATRIX__RANK_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/matrix/check_range.hpp>

namespace stan {
  namespace math {
   
    /**
     * Return the number of components of v less than v[s].
     *
     * @return Number of components of v less than v[s].
     * @tparam T Type of elements of the vector.
     */
    template <typename T>
    inline size_t rank(const std::vector<T> & v, int s) {
      size_t size = v.size();
      using stan::error_handling::check_range;
      check_range(size,s,"in the function rank(v,s)",s);
      s--;
      size_t count(0U);
      T compare(v[s]);
      for (size_t i = 0U; i < size; ++i)
        if (v[i]<compare) count++;
      return count;
    }

    /**
     * Return the number of components of v less than v[s].
     *
     * @return Number of components of v less than v[s].
     * @tparam T Type of elements of the vector.
     */
    template <typename T, int R, int C>
    inline size_t rank(const Eigen::Matrix<T,R,C> & v, int s) {
      size_t size = v.size();
      using stan::error_handling::check_range;
      check_range(size,s,"in the function rank(v,s)",s);
      s--;
      const T * vv = v.data();
      size_t count(0U);
      T compare(vv[s]);
      for (size_t i = 0U; i < size; ++i)
        if (vv[i]<compare) count++;
      return count;
    }
    
  }
}
#endif
