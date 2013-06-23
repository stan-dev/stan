// Arguments: Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/exponential.hpp>

#include <stan/math/functions/multiply_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsExponential : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(2);

    param[0] = 2.0;                 // y
    param[1] = 1.5;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-2.594535);  // expected log_prob

    param[0] = 15.0;                // y
    param[1] = 3.9;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-57.13902);  // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
    // beta
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_inv_scale, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_inv_scale>::type 
  log_prob(const T_y& y, const T_inv_scale& beta, 
       const T2&, const T3&, const T4&, const T5&,
       const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::exponential_log(y, beta);
  }

  template <bool propto, 
      typename T_y, typename T_inv_scale, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_inv_scale>::type 
  log_prob(const T_y& y, const T_inv_scale& beta, 
       const T2&, const T3&, const T4&, const T5&, 
       const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::exponential_log<propto>(y, beta);
  }
  
  
  template <typename T_y, typename T_inv_scale, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_inv_scale& beta, 
        const T2&, const T3&, const T4&, const T5&, 
        const T6&, const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;
    using stan::math::multiply_log;
    using boost::math::lgamma;
    using stan::prob::NEG_LOG_TWO_OVER_TWO;
    
    var logp(0);
    if (include_summand<true,T_inv_scale>::value)
      logp += log(beta);
    if (include_summand<true,T_y,T_inv_scale>::value)
      logp -= beta * y;
    return logp;
  }
};

TEST(ProbDistributionsExponential,Cumulative) {
  using std::numeric_limits;
  using stan::prob::exponential_cdf;
  EXPECT_FLOAT_EQ(0.95021293, exponential_cdf(2.0,1.5));
  EXPECT_FLOAT_EQ(1.0, exponential_cdf(15.0,3.9));
  EXPECT_FLOAT_EQ(0.62280765, exponential_cdf(0.25,3.9));

  // ??
  // EXPECT_FLOAT_EQ(0.0,
  //                 exponential_cdf(-numeric_limits<double>::infinity(),
  //                                 1.5));
  EXPECT_FLOAT_EQ(0.0, exponential_cdf(0.0,1.5));
  EXPECT_FLOAT_EQ(1.0,
                  exponential_cdf(numeric_limits<double>::infinity(),
                                  1.5));

}
