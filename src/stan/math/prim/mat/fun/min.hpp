#ifndef STAN_MATH_PRIM_MAT_FUN_MIN_HPP
#define STAN_MATH_PRIM_MAT_FUN_MIN_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace stan {
  namespace math {

    /**
     * Returns the minimum coefficient in the specified
     * column vector.
     * @param x Specified vector.
     * @return Minimum coefficient value in the vector.
     * @tparam Type of values being compared and returned
     */
    inline int min(const std::vector<int>& x) {
      if (x.size() == 0)
        throw std::domain_error("error: cannot take min of empty int vector");
      int min = x[0];
      for (size_t i = 1; i < x.size(); ++i)
        if (x[i] < min)
          min = x[i];
      return min;
    }

    /**
     * Returns the minimum coefficient in the specified
     * column vector.
     * @param x Specified vector.
     * @return Minimum coefficient value in the vector.
     * @tparam Type of values being compared and returned
     */
    template <typename T>
    inline T min(const std::vector<T>& x) {
      if (x.size() == 0)
        return std::numeric_limits<T>::infinity();
      T min = x[0];
      for (size_t i = 1; i < x.size(); ++i)
        if (x[i] < min)
          min = x[i];
      return min;
    }

    /**
     * Returns the minimum coefficient in the specified
     * matrix, vector, or row vector.
     * @param m Specified matrix, vector, or row vector.
     * @return Minimum coefficient value in the vector.
     */
    template <typename T, int R, int C>
    inline T min(const Eigen::Matrix<T, R, C>& m) {
      if (m.size() == 0)
        return std::numeric_limits<double>::infinity();
      return m.minCoeff();
    }

  }
}
#endif
