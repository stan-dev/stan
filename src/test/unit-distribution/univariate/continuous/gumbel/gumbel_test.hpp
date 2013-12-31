// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/gumbel.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionGumbel : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // beta
    parameters.push_back(param);
    log_prob.push_back(-1.0); // expected log_prob

    param[0] = 1;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // beta
    parameters.push_back(param);
    log_prob.push_back(-1.367879); // expected log_prob

    param[0] = -2;          // y
    param[1] = 0;           // mu
    param[2] = 1;           // beta
    parameters.push_back(param);
    log_prob.push_back(-5.389056); // expected log_prob

    param[0] = -3.5;          // y
    param[1] = 1.9;           // mu
    param[2] = 7.2;           // beta
    parameters.push_back(param);
    log_prob.push_back(-3.341081); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
    // mu
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
    value.push_back(-numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& beta,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::gumbel_log(y, mu, beta);
  }

  template <bool propto, 
      typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& beta,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::gumbel_log<propto>(y, mu, beta);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_loc& mu, const T_scale& beta,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;

    var lp(0.0);

    if (include_summand<true,T_scale>::value)
      lp -= log(beta);
    if (include_summand<true,T_y,T_loc,T_scale>::value)
      lp -= (y - mu) / beta + exp((mu - y) / beta);
    return lp;
  }
};

