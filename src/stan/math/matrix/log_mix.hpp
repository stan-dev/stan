#ifndef STAN__MATH__MATRIX__LOG_SUM_EXP_HPP
#define STAN__MATH__MATRIX__LOG_SUM_EXP_HPP

#include <limits>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/matrix/check_matching_sizes.hpp>
#include <stan/error_handling/matrix/check_simplex.hpp>
#include <stan/math/matrix/Eigen.hpp>


namespace stan {

  namespace math {

    /**
     * Return the log mixture density with specified mixing proportion
     * and log densities.
     *
     * \f[
     * \mbox{log\_mix}(\alpha, \lambda) 
     * = \log \left( \alpha_1 \lambda_1 
     * + \alpha_2 \lambda_2 + \ldots  + \alpha_n \lambda_n \right).
     * \f]
     * 
     * \f[
     * \frac{\partial}{\partial \alpha_i} 
     * \mbox{log\_mix}(\alpha, \lambda)
     * = FIXME
     * \f]
     *
     * \f[
     * \frac{\partial}{\partial \lambda_i} 
     * \mbox{log\_mix}(\alpha, \lambda)
     * = FIXME what to do about error handling in terms of vector sizes?
     * \f]
     *
     * @param[in] alpha mixing proportios in [0, 1], k-simplex
     * @param[in] lambda vector of log densities
     * @return log mixture of densities in specified proportion
     */
    double log_mix(const Eigen::Matrix<double,Eigen::Dynamic,1>& alpha,
                   const Eigen::Matrix<double,Eigen::Dynamic,1>& lambda) {
      using std::numeric_limits;
      using std::log;
      using std::exp;
      stan::math::check_not_nan("log_mix", "alpha", alpha);
      stan::math::check_not_nan("log_mix", "lambda", lambda);
      stan::math::check_simplex("log_mix", "alpha", alpha);
      stan::math::check_matching_sizes("log_mix", "alpha", alpha, "lambda", lambda);
      double max = -numeric_limits<double>::infinity();
      int d = lambda.size();
      for (int i = 0; i < d; ++i) 
        if (lambda(i) > max) 
          max = lambda(i);
      double sum = 0.0;
      for (int i = 0; i < d; ++i)
        if (-numeric_limits<double>::infinity() != lambda(i))
          sum += alpha(i) * exp(lambda(i) - max);
      return max + log(sum);
    }

  }
}

#endif
