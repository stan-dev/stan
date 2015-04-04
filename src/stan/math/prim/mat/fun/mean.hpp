#ifndef STAN_MATH_PRIM_MAT_FUN_MEAN_HPP
#define STAN_MATH_PRIM_MAT_FUN_MEAN_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/err/check_nonzero_size.hpp>
#include <boost/math/tools/promotion.hpp>
#include <vector>

namespace stan {
  namespace math {

    /**
     * Returns the sample mean (i.e., average) of the coefficients
     * in the specified standard vector.
     * @param v Specified vector.
     * @return Sample mean of vector coefficients.
     * @throws std::domain_error if the size of the vector is less
     * than 1.
     */
    template <typename T>
    inline
    typename boost::math::tools::promote_args<T>::type
    mean(const std::vector<T>& v) {
      stan::math::check_nonzero_size("mean", "v", v);
      T sum(v[0]);
      for (size_t i = 1; i < v.size(); ++i)
        sum += v[i];
      return sum / v.size();
    }

    /**
     * Returns the sample mean (i.e., average) of the coefficients
     * in the specified vector, row vector, or matrix.
     * @param m Specified vector, row vector, or matrix.
     * @return Sample mean of vector coefficients.
     */
    template <typename T, int R, int C>
    inline
    typename boost::math::tools::promote_args<T>::type
    mean(const Eigen::Matrix<T, R, C>& m) {
      stan::math::check_nonzero_size("mean", "m", m);
      return m.mean();
    }

  }
}
#endif
