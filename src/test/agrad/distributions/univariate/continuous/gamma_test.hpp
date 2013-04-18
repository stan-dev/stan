// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/gamma.hpp>

#include <stan/math/functions/multiply_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsGamma : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.0;                 // y
    param[1] = 2.0;                 // alpha
    param[2] = 2.0;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-0.6137056); // expected log_prob

    param[0] = 2.0;                 // y
    param[1] = 0.25;                // alpha
    param[2] = 0.75;                // beta
    parameters.push_back(param);
    log_prob.push_back(-3.379803);  // expected log_prob

    param[0] = 1.0;                 // y
    param[1] = 1.0;                 // alpha
    param[2] = 1.0;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-1.0);       // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
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

  template <typename T_y, typename T_shape, typename T_inv_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_shape, T_inv_scale>::type 
  log_prob(const T_y& y, const T_shape& alpha, const T_inv_scale& beta,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::gamma_log(y, alpha, beta);
  }

  template <bool propto, 
      typename T_y, typename T_shape, typename T_inv_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_shape, T_inv_scale>::type 
  log_prob(const T_y& y, const T_shape& alpha, const T_inv_scale& beta,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::gamma_log<propto>(y, alpha, beta);
  }
  
  
  template <typename T_y, typename T_shape, typename T_inv_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_shape& alpha, const T_inv_scale& beta,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;
    using stan::math::multiply_log;
    
    var lp(0);
    if (include_summand<true,T_shape>::value)
      lp -= lgamma(alpha);
    if (include_summand<true,T_shape,T_inv_scale>::value)
      lp += multiply_log(alpha,beta);
    if (include_summand<true,T_y,T_shape>::value)
      lp += multiply_log(alpha-1.0,y);
    if (include_summand<true,T_y,T_inv_scale>::value)
      lp -= beta * y;
    return lp;
  }
};
