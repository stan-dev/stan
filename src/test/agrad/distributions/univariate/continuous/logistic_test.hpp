// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/logistic.hpp>

#include <stan/math/functions/log1p.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsLogistic : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.2;           // y
    param[1] = 0.3;           // mu
    param[2] = 2.0;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-2.129645); // expected log_prob

    param[0] = -1.0;          // y
    param[1] = 0.2;           // mu
    param[2] = 0.25;          // sigma
    parameters.push_back(param);
    log_prob.push_back(-3.430098); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    
    // mu
    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // sigma
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::logistic_log(y, mu, sigma);
  }

  template <bool propto, 
      typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::logistic_log<propto>(y, mu, sigma);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
      using stan::prob::include_summand;
      using stan::math::log1p;
      var lp(0.0);
      if (include_summand<true,T_y,T_loc,T_scale>::value)
  lp -= (y - mu) / sigma;
      if (include_summand<true,T_scale>::value)
  lp -= log(sigma);
      if (include_summand<true,T_y,T_loc,T_scale>::value)
  lp -= 2.0 * log1p(exp(-(y - mu)/sigma));
      return lp;
  }
};


TEST(ProbDistributionsLogisticCDF, Values) {
    EXPECT_FLOAT_EQ(0.047191944, stan::prob::logistic_cdf(-3.45, 5.235, 2.89));
}
