#ifndef STAN__MATH__FUNCTIONS__MIX_HPP
#define STAN__MATH__FUNCTIONS__MIX_HPP

#include <cmath>
#include <stan/error_handling/scalar/check_bounded.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/math/functions/log_sum_exp.hpp>
#include <stan/math/functions/log1m.hpp>

namespace stan {

  namespace math {

    /**
     * Return the log mixture density with specified mixing proportion
     * and log densities.
     *
     * \f[
     * \mbox{log\_mix}(\theta, \lambda_1, \lambda_2) 
     * = \log \left( \theta \lambda_1 + (1 - \theta) \lambda_2 \right).
     * \f]
     * 
     * \f[
     * \frac{\partial}{\partial \theta} 
     * \mbox{log\_mix}(\theta, \lambda_1, \lambda_2)
     * = FIXME
     * \f]
     *
     * \f[
     * \frac{\partial}{\partial \lambda_1} 
     * \mbox{log\_mix}(\theta, \lambda_1, \lambda_2)
     * = FIXME
     * \f]
     *
     * \f[
     * \frac{\partial}{\partial \lambda_2} 
     * \mbox{log\_mix}(\theta, \lambda_1, \lambda_2)
     * = FIXME
     * \f]
     * 
     * @param[in] theta mixing proportion in [0,1].
     * @param lambda1 first log density.
     * @param lambda2 second log density.
     * @return log mixture of densities in specified proportion
     */
    double log_mix(double theta,
                   double lambda1,
                   double lambda2) {
      using std::log;
      stan::error_handling::check_not_nan("log_mix","lambda1",lambda1);
      stan::error_handling::check_not_nan("log_mix","lambda2",lambda2);
      stan::error_handling::check_bounded("log_mix","theta",theta,0,1);
      return log_sum_exp(log(theta) + lambda1,
                         log1m(theta) + lambda2);
    }

  }

}

#endif
