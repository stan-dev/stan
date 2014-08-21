#ifndef STAN__MATH__MATRIX__LOG_SOFTMAX_HPP
#define STAN__MATH__MATRIX__LOG_SOFTMAX_HPP

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/log_sum_exp.hpp>
#include <stan/math/error_handling/matrix/check_nonzero_size.hpp>

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
      using stan::math::log_sum_exp;
      stan::math::check_nonzero_size("log_softmax(%1%)",v,"v", (double*)0);
      Eigen::Matrix<T,Eigen::Dynamic,1> theta(v.size());
      T z = log_sum_exp(v);
      for (int i = 0; i < v.size(); ++i)
        theta(i) = v(i) - z;
      return theta;
      // T sum(0.0);
      // T max_v = v.maxCoeff();
      // for (int i = 0; i < v.size(); ++i)
      //   sum += exp(v(i) - max_v); // log_sum_exp trick
      // T log_sum = log(sum);
      // for (int i = 0; i < v.size(); ++i)
      //   theta(i) = (v(i) - max_v) - log_sum;
      // return theta;
    }

  }
}
#endif
