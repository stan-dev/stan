#ifndef STAN__MATH__MATRIX__FILL_HPP
#define STAN__MATH__MATRIX__FILL_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace math {

    /**
     * Fill the specified container with the specified value.
     *
     * This base case simply assigns the value to the container.
     *
     * @tparam T Type of reference container.
     * @tparam S Type of value.
     * @param x Container.
     * @param y Value.
     */
    template <typename T, typename S>
    void fill(T& x, const S& y) {
      x = y;
    }
    
    /**
     * Fill the specified container with the specified value.
     *
     * The specified matrix is filled by element.
     *
     * @tparam T Type of scalar for matrix container.
     * @tparam R Row type of matrix.
     * @tparam C Column type of matrix.
     * @tparam S Type of value.
     * @param x Container.
     * @param y Value.
     */
    template <typename T, int R, int C, typename S>
    void fill(Eigen::Matrix<T,R,C>& x, const S& y) {
      x.fill(y);
    }

    /**
     * Fill the specified container with the specified value.
     *
     * Each container in the specified standard vector is filled
     * recursively by calling <code>fill</code>.
     *
     * @tparam T Type of container in vector.
     * @tparam S Type of value.
     * @param x Container.
     * @param y Value.
     */
    template <typename T, typename S>
    void fill(std::vector<T>& x, const S& y) {
      for (size_t i = 0; i < x.size(); ++i)
        fill(x[i],y);
    }

    

  }
}
#endif
