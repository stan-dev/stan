// Arguments: Ints, Doubles
#include <stan/prob/distributions/univariate/discrete/bernoulli.hpp>

#include <stan/math/functions/logit.hpp>
#include <stan/math/functions/log1m.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsBernoulliLogistic : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& log_prob) {
    using stan::math::logit;
    using std::exp;
    vector<double> param(2);

    param[0] = 1;           // n
    param[1] = logit(0.25); // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.25)); // expected log_prob

    param[0] = 0;           // n
    param[1] = logit(0.25); // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.75)); // expected log_prob

    param[0] = 1;           // n
    param[1] = logit(0.01); // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.01)); // expected log_prob

    param[0] = 0;           // n
    param[1] = logit(0.01); // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.99)); // expected log_prob

    param[0] = 0;            // n
    param[1] = 25;           // theta
    parameters.push_back(param);
    log_prob.push_back(-25); // expected log_prob

    param[0] = 1;            // n
    param[1] = -25;          // theta
    parameters.push_back(param);
    log_prob.push_back(-25); // expected log_prob
    
    param[0] = 0;           // n
    param[1] = -25;         // theta
    parameters.push_back(param);
    log_prob.push_back(-exp(-25)); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1);

    index.push_back(0U);
    value.push_back(2);

    // theta
  }

  template <class T_n, class T_prob, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_n, T_prob>::type 
  log_prob(const T_n& n, const T_prob& theta, const T2&,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::bernoulli_logit_log(n, theta);
  }

  template <bool propto, 
      class T_n, class T_prob, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_n, T_prob>::type 
  log_prob(const T_n& n, const T_prob& theta, const T2&,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::bernoulli_logit_log<propto>(n, theta);
  }
  
  
  template <class T_n, class T_prob, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_n& n, const T_prob& theta, const T2&,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using std::log;
    using stan::math::log1m;
    using stan::prob::include_summand;

    if (include_summand<true,T_prob>::value) {
      T_prob ntheta = (2*n-1) * theta;
      // Handle extreme values gracefully using Taylor approximations.
      const static double cutoff = 20.0;
      if (ntheta > cutoff)
  return -exp(-ntheta);
      else if (ntheta < -cutoff)
  return ntheta;
      else
  return -log(1 + exp(-ntheta));
    }
    return 0.0;
  }
};
