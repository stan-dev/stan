#ifndef STAN_MATH_PRIM_MAT_FUN_LOG_SOFTMAX_HPP
#define STAN_MATH_PRIM_MAT_FUN_LOG_SOFTMAX_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/log_sum_exp.hpp>
#include <stan/math/prim/scal/err/check_nonzero_size.hpp>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace stan {
  namespace math {

   /**
     * Return the natural logarithm of the softmax of the specified
     * vector.
     *
     * \f$
     * \log \mbox{softmax}(y)
     * \ = \ y - \log \sum_{k=1}^K \exp(y_k)
     * \ = \ y - \mbox{log\_sum\_exp}(y).
     * \f$
     *
     * For the log softmax function, the entries in the Jacobian are
     * \f$
     * \frac{\partial}{\partial y_m} \mbox{softmax}(y)[k]
     * = \left\{
     * \begin{array}{ll}
     * 1 - \mbox{softmax}(y)[m]
     * & \mbox{ if } m = k, \mbox{ and}
     * \\[6pt]
     * \mbox{softmax}(y)[m]
     * & \mbox{ if } m \neq k.
     * \end{array}
     * \right.
     * \f$
     *
     * @tparam T Scalar type of values in vector.
     * @param[in] v Vector to transform.
     * @return Unit simplex result of the softmax transform of the vector.
     */
    template <typename T>
    inline Eigen::Matrix<T, Eigen::Dynamic, 1>
    log_softmax(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) {
      using std::exp;
      using std::log;
      using stan::math::log_sum_exp;
      stan::math::check_nonzero_size("log_softmax", "v", v);
      Eigen::Matrix<T, Eigen::Dynamic, 1> theta(v.size());
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
