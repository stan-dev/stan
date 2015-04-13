#ifndef STAN_MATH_PRIM_MAT_FUN_MAX_HPP
#define STAN_MATH_PRIM_MAT_FUN_MAX_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace stan {
  namespace math {

    /**
     * Returns the maximum coefficient in the specified
     * column vector.
     * @param x Specified vector.
     * @return Maximum coefficient value in the vector.
     * @tparam Type of values being compared and returned
     * @throw std::domain_error If the size of the vector is zero.
     */
    inline int max(const std::vector<int>& x) {
      if (x.size() == 0)
        throw std::domain_error("error: cannot take max of empty int vector");
      int max = x[0];
      for (size_t i = 1; i < x.size(); ++i)
        if (x[i] > max)
          max = x[i];
      return max;
    }

    /**
     * Returns the maximum coefficient in the specified
     * column vector.
     * @param x Specified vector.
     * @return Maximum coefficient value in the vector.
     * @tparam T Type of values being compared and returned
     */
    template <typename T>
    inline T max(const std::vector<T>& x) {
      if (x.size() == 0)
        return -std::numeric_limits<T>::infinity();
      T max = x[0];
      for (size_t i = 1; i < x.size(); ++i)
        if (x[i] > max)
          max = x[i];
      return max;
    }

    /**
     * Returns the maximum coefficient in the specified
     * vector, row vector, or matrix.
     * @param m Specified vector, row vector, or matrix.
     * @return Maximum coefficient value in the vector.
     */
    template <typename T, int R, int C>
    inline T max(const Eigen::Matrix<T, R, C>& m) {
      if (m.size() == 0)
        return -std::numeric_limits<double>::infinity();
      return m.maxCoeff();
    }

  }
}
#endif
