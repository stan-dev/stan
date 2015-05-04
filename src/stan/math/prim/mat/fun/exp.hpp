#ifndef STAN_MATH_PRIM_MAT_FUN_EXP_HPP
#define STAN_MATH_PRIM_MAT_FUN_EXP_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <limits>

namespace stan {
  namespace math {

    /**
     * Return the element-wise exponentiation of the matrix or vector.
     *
     * @param m The matrix or vector.
     * @return ret(i, j) = exp(m(i, j))
     */
    template<typename T, int Rows, int Cols>
    inline Eigen::Matrix<T, Rows, Cols>
    exp(const Eigen::Matrix<T, Rows, Cols>& m) {
      return m.array().exp().matrix();
    }

    // FIXME:
    // specialization not needed once Eigen fixes issue:
    // http:// eigen.tuxfamily.org/bz/show_bug.cgi?id=859
    template<int Rows, int Cols>
    inline Eigen::Matrix<double, Rows, Cols>
    exp(const Eigen::Matrix<double, Rows, Cols>& m) {
      Eigen::Matrix<double, Rows, Cols> mat = m.array().exp().matrix();
      for (int i = 0, size_ = mat.size(); i < size_; i++)
        if (boost::math::isnan(m(i)))
          mat(i) = std::numeric_limits<double>::quiet_NaN();
      return mat;
    }

  }
}
#endif
