// Arguments: Ints, Doubles, Doubles
#include <stan/prob/distributions/univariate/discrete/neg_binomial_2.hpp>
#include <stan/prob/distributions/univariate/discrete/binomial.hpp>

#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/log_sum_exp.hpp>
#include <stan/math/functions/binomial_coefficient_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsNegBinomial2Log : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 10;           // n
    param[1] = 2.0;          // eta
    param[2] = 1.5;          // phi
    parameters.push_back(param);
    log_prob.push_back(-3.208872); // expected log_prob

    param[0] = 100;          // n
    param[1] = -3.0;          // eta
    param[2] = 3.5;          // phi
    parameters.push_back(param);
    log_prob.push_back(-416.3829); // expected log_prob

    param[0] = 100;          // n
    param[1] = -10;          // eta
    param[2] = 200;          // phi
    parameters.push_back(param);
    log_prob.push_back(-1342.303); // expected log_prob
  }

  void invalid_values(vector<size_t>& index,
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);

    // phi
    index.push_back(2U);
    value.push_back(-1);
  }

  template <class T_n, class T_log_location, class T_inv_scale,
            typename T3, typename T4, typename T5,
            typename T6, typename T7, typename T8,
            typename T9>
  typename stan::return_type<T_log_location,T_inv_scale>::type
  log_prob(const T_n& n, const T_log_location& eta, const T_inv_scale& phi,
     const T3&, const T4&, const T5&,
     const T6&, const T7&, const T8&,
     const T9&) {
    return stan::prob::neg_binomial_2_log_log(n, eta, phi);
  }

  template <bool propto,
      class T_n, class T_log_location, class T_inv_scale,
            typename T3, typename T4, typename T5,
            typename T6, typename T7, typename T8,
            typename T9>
  typename stan::return_type<T_log_location,T_inv_scale>::type
  log_prob(const T_n& n, const T_log_location& eta, const T_inv_scale& phi,
     const T3&, const T4&, const T5&,
     const T6&, const T7&, const T8&,
     const T9&) {
    return stan::prob::neg_binomial_2_log_log<propto>(n, eta, phi);
  }


  template <class T_n, class T_log_location, class T_inv_scale,
            typename T3, typename T4, typename T5,
            typename T6, typename T7, typename T8,
            typename T9>
  var log_prob_function(const T_n& n, const T_log_location& eta, const T_inv_scale& phi,
      const T3&, const T4&, const T5&,
      const T6&, const T7&, const T8&,
      const T9&) {
    using std::log;
    using stan::math::binomial_coefficient_log;
    using stan::math::log_sum_exp;
    using stan::math::multiply_log;
    using stan::prob::include_summand;

    var logp(0);
    // Special case where negative binomial reduces to Poisson
    /*if (eta > 1e10) {
      if (include_summand<true>::value)
        logp -= lgamma(n + 1.0);
      if (include_summand<true,T_log_location>::value ||
          include_summand<true,T_inv_scale>::value) {
        typename stan::return_type<T_log_location, T_inv_scale>::type lambda;
        lambda = eta / phi;
        logp += multiply_log(n, lambda) - lambda;
      }
      return logp;
    } */
    // More typical cases
    if (include_summand<true,T_inv_scale>::value)
      if (n != 0)
        logp += binomial_coefficient_log<typename stan::scalar_type<T_inv_scale>::type>
          (n + phi - 1.0, n);
    if (include_summand<true,T_log_location,T_inv_scale>::value)
      logp += n*eta + multiply_log(phi,phi) - (n+phi)*log_sum_exp(eta,log(phi));
    return logp;
  }
};

