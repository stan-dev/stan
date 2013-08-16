#ifndef __STAN__MATH__MATRIX__VARIANCE_HPP__
#define __STAN__MATH__MATRIX__VARIANCE_HPP__

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/mean.hpp>

namespace stan {
  namespace math {
    
    /**
     * Returns the sample variance (divide by length - 1) of the
     * coefficients in the specified standard vector.
     * @param v Specified vector.
     * @return Sample variance of vector.
     * @throws std::domain_error if the size of the vector is less
     * than 1.
     */
    template <typename T>
    inline 
    typename boost::math::tools::promote_args<T>::type
    variance(const std::vector<T>& v) {
      validate_nonzero_size(v,"variance");
      if (v.size() == 1)
        return 0.0;
      T v_mean(mean(v));
      T sum_sq_diff(0);
      for (size_t i = 0; i < v.size(); ++i) {
        T diff = v[i] - v_mean;
        sum_sq_diff += diff * diff;
      }
      return sum_sq_diff / (v.size() - 1);
    }

    /**
     * Returns the sample variance (divide by length - 1) of the
     * coefficients in the specified column vector.
     * @param m Specified vector.
     * @return Sample variance of vector.
     */
    template <typename T, int R, int C>
    inline
    typename boost::math::tools::promote_args<T>::type
    variance(const Eigen::Matrix<T,R,C>& m) {
      validate_nonzero_size(m,"variance");
      if (m.size() == 1)
        return 0.0;
      typename boost::math::tools::promote_args<T>::type 
        mn(mean(m));
      typename boost::math::tools::promote_args<T>::type 
        sum_sq_diff(0);
      for (int i = 0; i < m.size(); ++i) {
        typename boost::math::tools::promote_args<T>::type 
          diff = m(i) - mn;
        sum_sq_diff += diff * diff;
      }
      return sum_sq_diff / (m.size() - 1);
    }
    
  }
}
#endif
