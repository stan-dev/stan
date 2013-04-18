// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/lognormal.hpp>

#include <stan/math/functions/square.hpp>
#include <stan/math/constants.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsLognormal : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.2;           // y
    param[1] = 0.3;           // mu
    param[2] = 1.5;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.509802579); // expected log_prob

    param[0] = 12.0;          // y
    param[1] = 3.0;           // mu
    param[2] = 0.9;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-3.462263161); // expected log_prob
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
    return stan::prob::lognormal_log(y, mu, sigma);
  }

  template <bool propto, 
      typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::lognormal_log<propto>(y, mu, sigma);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;
    using stan::math::pi;
    using stan::math::square;
    using stan::prob::NEG_LOG_SQRT_TWO_PI;
      
    var lp(0.0);
    if (include_summand<true>::value)
      lp += NEG_LOG_SQRT_TWO_PI;
    if (include_summand<true,T_scale>::value)
      lp -= log(sigma);
    if (include_summand<true,T_y>::value)
      lp -= log(y);
    if (include_summand<true,T_y,T_loc,T_scale>::value)
      lp -= square(log(y) - mu) / (2.0 * sigma * sigma);
    return lp;
  }
};



TEST(ProbDistributionsLognormal,Cumulative) {
  using stan::prob::lognormal_cdf;
  EXPECT_FLOAT_EQ(0.4687341, lognormal_cdf(1.2,0.3,1.5));
  EXPECT_FLOAT_EQ(0.2835506, lognormal_cdf(12.0,3.0,0.9));

  double pos_inf = std::numeric_limits<double>::infinity();
 
  // ?? double neg_inf = -pos_inf;
  // ?? EXPECT_FLOAT_EQ(0.0,lognormal_cdf(neg_inf,0.0,1.0));

  EXPECT_FLOAT_EQ(0.0,lognormal_cdf(0.0,0.0,1.0));
  EXPECT_FLOAT_EQ(1.0,lognormal_cdf(pos_inf,0.0,1.0));
}

