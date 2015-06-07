#ifndef STAN_MATH_REV_SCAL_FUN_LOG_MIX_HPP
#define STAN_MATH_REV_SCAL_FUN_LOG_MIX_HPP

#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/log_mix.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <cmath>

namespace stan {

  namespace math {

    /* Computes shared terms in log_mix partial derivative calculations
     *
     * @param[in] theta_val value of mixing proportion theta.
     * @param[in] lambda1_val value of log density multiplied by theta.
     * @param[in] lambda2_val value of log density multiplied by 1 - theta.
     * @param[out] one_m_exp_lam2_m_lam1 shared term in deriv calculation.
     * @param[out] one_m_t_prod_exp_lam2_m_lam1 shared term in deriv calculation.
     * @param[out] one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1 shared term in deriv calculation.
     */
    inline void
    log_mix_partial_helper(const double& theta_val,
                           const double& lambda1_val,
                           const double& lambda2_val,
                           double& one_m_exp_lam2_m_lam1,
                           double& one_m_t_prod_exp_lam2_m_lam1,
                           double& one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1) {
        using std::exp;
        double lam2_m_lam1 = lambda2_val - lambda1_val;
        double exp_lam2_m_lam1 = exp(lam2_m_lam1);
        one_m_exp_lam2_m_lam1 = 1 - exp_lam2_m_lam1;
        double one_m_t = 1 - theta_val;
        one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1
          = 1 / (theta_val + one_m_t_prod_exp_lam2_m_lam1);
    }

    /**
     * Return the log mixture density with specified mixing proportion
     * and log densities and its derivative at each.
     *
     * \f[
     * \mbox{log\_mix}(\theta, \lambda_1, \lambda_2)
     * = \log \left( \theta \exp(\lambda_1) + (1 - \theta) \exp(\lambda_2) \right).
     * \f]
     *
     * \f[
     * \frac{\partial}{\partial \theta}
     * \mbox{log\_mix}(\theta, \lambda_1, \lambda_2)
     * = \dfrac{\exp(\lambda_1) - \exp(\lambda_2)}
     * {\left( \theta \exp(\lambda_1) + (1 - \theta) \exp(\lambda_2) \right)}
     * \f]
     *
     * \f[
     * \frac{\partial}{\partial \lambda_1}
     * \mbox{log\_mix}(\theta, \lambda_1, \lambda_2)
     * = \dfrac{\theta \exp(\lambda_1)}
     * {\left( \theta \exp(\lambda_1) + (1 - \theta) \exp(\lambda_2) \right)}
     * \f]
     *
     * \f[
     * \frac{\partial}{\partial \lambda_2}
     * \mbox{log\_mix}(\theta, \lambda_1, \lambda_2)
     * = \dfrac{\theta \exp(\lambda_2)}
     * {\left( \theta \exp(\lambda_1) + (1 - \theta) \exp(\lambda_2) \right)}
     * \f]
     *
     * @tparam T_theta theta scalar type.
     * @tparam T_lambda1 lambda1 scalar type.
     * @tparam T_lambda2 lambda2 scalar type.
     *
     * @param[in] theta mixing proportion in [0, 1].
     * @param[in] lambda1 first log density.
     * @param[in] lambda2 second log density.
     * @return log mixture of densities in specified proportion
     */
    template <typename T_theta,
              typename T_lambda1,
              typename T_lambda2>
    inline
    typename return_type<T_theta, T_lambda1, T_lambda2>::type
    log_mix(const T_theta& theta,
            const T_lambda1& lambda1,
            const T_lambda2& lambda2) {
      using std::log;
      using stan::math::log_mix;
      using stan::math::log1m;
      using stan::is_constant_struct;

      OperandsAndPartials<T_theta, T_lambda1, T_lambda2>
        operands_and_partials(theta, lambda1, lambda2);

      double theta_double = value_of(theta);
      const double lambda1_double = value_of(lambda1);
      const double lambda2_double = value_of(lambda2);

      double log_mix_function_value
        = log_mix(theta_double, lambda1_double, lambda2_double);

      double one_m_exp_lam2_m_lam1(0.0);
      double one_m_t_prod_exp_lam2_m_lam1(0.0);
      double one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1(0.0);

      if (lambda1 > lambda2) {
        log_mix_partial_helper(theta_double,
                                   lambda1_double,
                                   lambda2_double,
                                   one_m_exp_lam2_m_lam1,
                                   one_m_t_prod_exp_lam2_m_lam1,
                                   one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1);
      } else {
        log_mix_partial_helper(1.0 - theta_double,
                                   lambda2_double,
                                   lambda1_double,
                                   one_m_exp_lam2_m_lam1,
                                   one_m_t_prod_exp_lam2_m_lam1,
                                   one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1);
        one_m_exp_lam2_m_lam1 = -one_m_exp_lam2_m_lam1;
        theta_double = one_m_t_prod_exp_lam2_m_lam1;
        one_m_t_prod_exp_lam2_m_lam1 = 1.0 - value_of(theta);
      }

      if (!is_constant_struct<T_theta>::value)
        operands_and_partials.d_x1[0]
          = one_m_exp_lam2_m_lam1
          * one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1;
      if (!is_constant_struct<T_lambda1>::value)
        operands_and_partials.d_x2[0]
          = theta_double
          * one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1;
      if (!is_constant_struct<T_lambda2>::value)
        operands_and_partials.d_x3[0]
          = one_m_t_prod_exp_lam2_m_lam1
          * one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1;

      return operands_and_partials.to_var(log_mix_function_value);
    }

  }  // namespace math

}  // namespace stan

#endif
