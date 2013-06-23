// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/double_exponential.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsDoubleExponential : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);
    
    param[0] = 1.0;                  // y
    param[1] = 1.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.6931472);  // expected log_prob

    param[0] = 2.0;                  // y
    param[1] = 1.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.693147);   // expected log_prob
    
    param[0] = -3.0;                 // y
    param[1] = 2.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-5.693147);   // expected log_prob
    
    param[0] = 1.0;                  // y
    param[1] = 0.0;                  // mu
    param[2] = 2.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.886294);   // expected log_prob

    param[0] = 1.9;                  // y
    param[1] = 2.3;                  // mu
    param[2] = 0.5;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.8);        // expected log_prob

    param[0] = 1.9;                  // y
    param[1] = 2.3;                  // mu
    param[2] = 0.25;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.9068528);        // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(0U);
    value.push_back(-numeric_limits<double>::infinity());
    
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
    return stan::prob::double_exponential_log(y, mu, sigma);
  }

  template <bool propto, 
      typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::double_exponential_log<propto>(y, mu, sigma);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using std::log;
    using std::fabs;
    using stan::prob::include_summand;
    using stan::prob::NEG_LOG_TWO;

    var logp(0);
    
    if (include_summand<true>::value)
      logp += NEG_LOG_TWO;
    if (include_summand<true,T_scale>::value)
      logp -= log(sigma);
    if (include_summand<true,T_y,T_loc,T_scale>::value)
      logp -= fabs(y - mu) / sigma;
    return logp;
  }
};


TEST(ProbDistributionsDoubleExponential,Cumulative) {
  EXPECT_FLOAT_EQ(0.5, stan::prob::double_exponential_cdf(1.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.8160603, stan::prob::double_exponential_cdf(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.003368973, stan::prob::double_exponential_cdf(-3.0,2.0,1.0));
  EXPECT_FLOAT_EQ(0.6967347, stan::prob::double_exponential_cdf(1.0,0.0,2.0));
  EXPECT_FLOAT_EQ(0.2246645, stan::prob::double_exponential_cdf(1.9,2.3,0.5));
  EXPECT_FLOAT_EQ(0.10094826, stan::prob::double_exponential_cdf(1.9,2.3,0.25));
}

