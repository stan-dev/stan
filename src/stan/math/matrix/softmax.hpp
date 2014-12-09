#ifndef STAN__MATH__MATRIX__SOFTMAX_HPP
#define STAN__MATH__MATRIX__SOFTMAX_HPP

#include <stan/error_handling/matrix/check_nonzero_size.hpp>
#include <cmath>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

   /**
     * Return the softmax of the specified vector.
     *
     * <p>
     * \f$
     * \mbox{softmax}(y)
     * = \frac{\exp(y)}
     * {\sum_{k=1}^K \exp(y_k)},
     * \f$
     *
     * <p>The entries in the Jacobian of the softmax function are given by
     * \f$
     * \begin{array}{l}
     * \displaystyle
     * \frac{\partial}{\partial y_m} \mbox{softmax}(y)[k]
     * \\[8pt]
     * \displaystyle
     * \mbox{ } \ \ \ = \left\{ 
     * \begin{array}{ll}
     * \mbox{softmax}(y)[k] - \mbox{softmax}(y)[k] \times \mbox{softmax}(y)[m]
     * & \mbox{ if } m = k, \mbox{ and}
     * \\[6pt]
     * \mbox{softmax}(y)[k] * \mbox{softmax}(y)[m]
     * & \mbox{ if } m \neq k.
     * \end{array}
     * \right.
     * \end{array}
     * \f$
     *
     * @tparam T Scalar type of values in vector.
     * @param[in] v Vector to transform.
     * @return Unit simplex result of the softmax transform of the vector.
     */
    template <typename T>
    inline Eigen::Matrix<T,Eigen::Dynamic,1>
    softmax(const Eigen::Matrix<T,Eigen::Dynamic,1>& v) {
      using std::exp;
      stan::error_handling::check_nonzero_size("softmax", "v", v);
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
