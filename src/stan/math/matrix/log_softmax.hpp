#ifndef __STAN__MATH__MATRIX__LOG_SOFTMAX_HPP__
#define __STAN__MATH__MATRIX__LOG_SOFTMAX_HPP__

#include <cmath>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_nonzero_size.hpp>

namespace stan {
  namespace math {

   /**
     * Return the natural logarithm of the softmax of the specified vector.
     *
     * @tparam T Scalar type of values in vector.
     * @param[in] v Vector to transform.
     * @return Unit simplex result of the softmax transform of the vector.
     */
    template <typename T>
    inline Eigen::Matrix<T,Eigen::Dynamic,1>
    log_softmax(const Eigen::Matrix<T,Eigen::Dynamic,1>& v) {
      using std::exp;
      using std::log;
      stan::math::validate_nonzero_size(v,"vector softmax");
      Eigen::Matrix<T,Eigen::Dynamic,1> theta(v.size());
      T sum(0.0);
      T max_v = v.maxCoeff();
      for (int i = 0; i < v.size(); ++i)
        sum += exp(v(i) - max_v); // log_sum_exp trick
      T log_sum = log(sum);
      for (int i = 0; i < v.size(); ++i)
        theta(i) = (v(i) - max_v) - log_sum;
      return theta;
    }

  }
}
#endif
