// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/cauchy.hpp>

#include <stan/math/functions/log1p.hpp>
#include <stan/math/functions/square.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsCauchy : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);
    
    param[0] = 1.0;                // y
    param[1] = 0.0;                // mu
    param[2] = 1.0;                // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.837877); // expected log_prob

    param[0] = -1.5;                // y
    param[1] = 0.0;                 // mu
    param[2] = 1.0;                 // sigma
    parameters.push_back(param);
    log_prob.push_back(-2.323385); // expected log_prob

    param[0] = -1.5;                // y
    param[1] = -1.0;                // mu
    param[2] = 1.0;                 // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.367873); // expected log_prob
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
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::cauchy_log(y, mu, sigma);
  }

  template <bool propto, 
      typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::cauchy_log<propto>(y, mu, sigma);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::math::log1p;
    using stan::math::square;
    using stan::prob::include_summand;
    
    var lp = 0.0;
    if (include_summand<true>::value)
      lp += stan::prob::NEG_LOG_PI;
    if (include_summand<true,T_scale>::value)
      lp -= log(sigma);
    if (include_summand<true,T_y,T_loc,T_scale>::value)
      lp -= log1p(square((y - mu) / sigma));
    return lp;
  }
};


TEST(ProbDistributionsCauchy,Cumulative) {
  using stan::prob::cauchy_cdf;
  EXPECT_FLOAT_EQ(0.75, cauchy_cdf(1.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.187167, cauchy_cdf(-1.5, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.187167, cauchy_cdf(-2.5, -1.0, 1.0));
}
