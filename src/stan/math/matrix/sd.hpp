#ifndef STAN__MATH__MATRIX__SD_HPP
#define STAN__MATH__MATRIX__SD_HPP

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/variance.hpp>
#include <stan/error_handling/matrix/check_nonzero_size.hpp>

namespace stan {
  namespace math {
    
    /**
     * Returns the unbiased sample standard deviation of the
     * coefficients in the specified column vector.
     * @param v Specified vector.
     * @return Sample variance of vector.
     */
    template <typename T>
    inline 
    typename boost::math::tools::promote_args<T>::type
    sd(const std::vector<T>& v) {
      stan::error_handling::check_nonzero_size("sd", "v", v);    
      if (v.size() == 1) return 0.0;
      return sqrt(variance(v));
    }

    /**
     * Returns the unbiased sample standard deviation of the
     * coefficients in the specified vector, row vector, or matrix.
     * @param m Specified vector, row vector or matrix.
     * @return Sample variance.
     */
    template <typename T, int R, int C>
    inline 
    typename boost::math::tools::promote_args<T>::type
    sd(const Eigen::Matrix<T,R,C>& m) {
      // FIXME: redundant with test in variance; second line saves sqrt
      stan::error_handling::check_nonzero_size("sd", "m", m);    
      if (m.size() == 1) return 0.0;
      return sqrt(variance(m));
    }    
    
  }
}
#endif
