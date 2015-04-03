#ifndef STAN_MATH_PRIM_SCAL_FUN_BINARY_LOG_LOSS_HPP
#define STAN_MATH_PRIM_SCAL_FUN_BINARY_LOG_LOSS_HPP

#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the log loss function for binary classification
     * with specified reference and response values.
     *
     * The log loss function for prediction \f$\hat{y} \in [0, 1]\f$
     * given outcome \f$y \in \{ 0, 1 \}\f$ is
     *
     * \f$\mbox{logloss}(1, \hat{y}) = -\log \hat{y} \f$, and
     *
     * \f$\mbox{logloss}(0, \hat{y}) = -\log (1 - \hat{y}) \f$.
     *
     * @param y Reference value in { 0 , 1 }.
     * @param y_hat Response value in [0, 1].
     * @return Log loss for response given reference value.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    binary_log_loss(const int y, const T y_hat) {
      using std::log;
      return -log(y ? y_hat : (1.0 - y_hat));
    }

  }
}

#endif
