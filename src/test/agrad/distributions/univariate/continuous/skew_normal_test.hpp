// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/skew_normal.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionSkewNormal : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(4);

    param[0] = 0.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 1.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-0.91893852); // expected log_prob

    param[0] = 1.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 1.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-0.898545); // expected log_prob

    param[0] = -2.0;          // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 2.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-12.585893); // expected log_prob

    param[0] = -3.5;          // y
    param[1] = 1.9;           // mu
    param[2] = 7.2;           // sigma
    param[3] = 2.9;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-6.6932335); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
    // mu
    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    // sigma
    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(0.0);

    //alpha
    index.push_back(3U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(3U);
    value.push_back(numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_loc, typename T_scale,
      typename T_shape, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale,T_shape>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T_shape& alpha, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::skew_normal_log(y, mu, sigma, alpha);
  }

  template <bool propto, 
      typename T_y, typename T_loc, typename T_scale,
      typename T_shape, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale, T_shape>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T_shape& alpha, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::skew_normal_log<propto>(y, mu, sigma, alpha);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T_shape, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T_shape& alpha, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;

    var logp(0.0);

    if (include_summand<true>::value)
      logp -=  0.5 * log(2.0 * boost::math::constants::pi<double>());
    if (include_summand<true, T_scale>::value)
      logp -= log(sigma);
    if (include_summand<true,T_y, T_loc, T_scale>::value)
      logp -= (y - mu) / sigma * (y - mu) / sigma * 0.5;
    if (include_summand<true,T_y,T_loc,T_scale,T_shape>::value)
      logp += log(erfc(-alpha * (y - mu) / (sigma * std::sqrt(2.0))));
    return logp;
  }
};

