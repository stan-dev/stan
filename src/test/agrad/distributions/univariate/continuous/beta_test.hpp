// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/beta.hpp>

#include <stan/math/functions/log1m.hpp>
#include <stan/math/functions/value_of.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsBeta : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 0.2;           // y
    param[1] = 1.0;           // alpha
    param[2] = 1.0;           // beta
    parameters.push_back(param);
    log_prob.push_back(0.0); // expected log_prob

    param[0] = 0.3;           // y
    param[1] = 12.0;          // alpha
    param[2] = 25.0;          // beta
    parameters.push_back(param);
    log_prob.push_back(1.628758); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1.0);
    
    index.push_back(0U);
    value.push_back(2.0);

    // alpha
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // beta
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_scale1, typename T_scale2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_scale1, T_scale2>::type 
  log_prob(const T_y& y, const T_scale1& alpha, const T_scale2& beta,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::beta_log(y, alpha, beta);
  }

  template <bool propto, 
      typename T_y, typename T_scale1, typename T_scale2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_scale1, T_scale2>::type 
  log_prob(const T_y& y, const T_scale1& alpha, const T_scale2& beta,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::beta_log<propto>(y, alpha, beta);
  }
  
  
  template <typename T_y, typename T_scale1, typename T_scale2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_scale1& alpha, const T_scale2& beta,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using std::log;
    using stan::math::log1m;
    using stan::math::value_of;
    using stan::prob::include_summand;
    
    var logp(0);
    
    if (include_summand<true,T_y,T_scale1>::value)
      logp += (alpha - 1.0) * log(y);
    if (include_summand<true,T_y,T_scale2>::value)
      logp += (beta - 1.0) * log1m(y);
    if (include_summand<true,T_scale1,T_scale2>::value)
      logp += lgamma(alpha + beta);
    if (include_summand<true,T_scale1>::value)
      logp -= lgamma(alpha);
    if (include_summand<true,T_scale2>::value)
      logp -= lgamma(beta);
    return logp;
  }
};
