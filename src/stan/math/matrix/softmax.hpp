#ifndef STAN__MATH__MATRIX__SOFTMAX_HPP
#define STAN__MATH__MATRIX__SOFTMAX_HPP

#include <stan/math/error_handling/matrix/check_nonzero_size.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <cmath>
#include <complex>

#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/util/Constants.h"

namespace stan {
  namespace math {

   /**
     * Return the softmax of the specified vector.
     *
     * @tparam T Scalar type of values in vector.
     * @param[in] v Vector to transform.
     * @return Unit simplex result of the softmax transform of the vector.
     */
    template <typename T>
    inline Eigen::Matrix<T,Eigen::Dynamic,1>
    softmax(const Eigen::Matrix<T,Eigen::Dynamic,1>& v) {
      using std::exp;
      stan::math::check_nonzero_size("softmax(%1%)",v,"v",(double*)0);
      Eigen::Matrix<T,Eigen::Dynamic,1> theta(v.size());
      T sum(0.0);
      T max_v = v.maxCoeff();
      for (int i = 0; i < v.size(); ++i) {
        theta(i) = exp(v(i) - max_v); // extra work for (v[i] == max_v)
        sum += theta(i);              // extra work vs. sum() w. auto-diff
      }
      for (int i = 0; i < v.size(); ++i)
        theta(i) /= sum;
      return theta;
    }

  }
}
#endif
