// Arguments: Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/rayleigh.hpp>

#include <stan/math/constants.hpp>
#include <stan/math/functions/square.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionRayleigh : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(2);

    param[0] = 4;           // y
    param[1] = 1;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-6.613705639); // expected log_prob

    param[0] = 1;           // y
    param[1] = 1;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.5); // expected log_prob

    param[0] = 2;          // y
    param[1] = 1;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.306852819); // expected log_prob

    param[0] = 3.5;          // y
    param[1] = 7.2;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-2.8135512); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {    
    // y
    index.push_back(0U);
    value.push_back(-1.0);

    // sigma
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_scale, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_scale>::type 
  log_prob(const T_y& y, const T_scale& sigma, const T2&,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::rayleigh_log(y, sigma);
  }

  template <bool propto, 
      typename T_y, typename T_scale, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_scale>::type 
  log_prob(const T_y& y, const T_scale& sigma, const T2&,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::rayleigh_log<propto>(y, sigma);
  }
  
  
  template <typename T_y, typename T_scale, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_scale& sigma, const T2&,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;
    using stan::math::pi;
    using stan::math::square;
    var lp(0.0);
    if (include_summand<true,T_y,T_scale>::value)
      lp -= 0.5 * y * y / (sigma * sigma);
    if (include_summand<true,T_scale>::value)
      lp -= 2.0 * log(sigma);
    if (include_summand<true, T_y>::value)
      lp += log(y);
    return lp;
  }
};

