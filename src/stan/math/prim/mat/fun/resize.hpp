#ifndef STAN_MATH_PRIM_MAT_FUN_RESIZE_HPP
#define STAN_MATH_PRIM_MAT_FUN_RESIZE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <vector>

namespace stan {
  namespace math {

    namespace {

      template <typename T>
      void resize(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& x,
                  const std::vector<size_t>& dims,
                  size_t pos) {
        x.resize(dims[pos], dims[pos+1]);
      }

      template <typename T>
      void resize(Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
                  const std::vector<size_t>& dims,
                  size_t pos) {
        x.resize(dims[pos]);
      }

      template <typename T>
      void resize(Eigen::Matrix<T, 1, Eigen::Dynamic>& x,
                  const std::vector<size_t>& dims,
                  size_t pos) {
        x.resize(dims[pos]);
      }

      template <typename T>
      void resize(T /*x*/,
                  const std::vector<size_t>& /*dims*/,
                  size_t /*pos*/) {
        // no-op
      }

      template <typename T>
      void resize(std::vector<T>& x,
                  const std::vector<size_t>& dims,
                  size_t pos) {
        x.resize(dims[pos]);
        ++pos;
        if (pos >= dims.size()) return;  // skips lowest loop to scalar
        for (size_t i = 0; i < x.size(); ++i)
          resize(x[i], dims, pos);
      }

    }

    /**
     * Recursively resize the specified vector of vectors,
     * which must bottom out at scalar values, Eigen vectors
     * or Eigen matrices.
     *
     * @param x Array-like object to resize.
     * @param dims New dimensions.
     * @tparam T Type of object being resized.
     */
    template <typename T>
    inline void resize(T& x, std::vector<size_t> dims) {
      resize(x, dims, 0U);
    }

  }
}
#endif
