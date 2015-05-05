// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/weibull_log.hpp>
#include <stan/math/prim/scal/prob/weibull_cdf.hpp>

#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsWeibull : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 2.0;                 // y
    param[1] = 1.0;                 // alpha
    param[2] = 1.0;                 // sigma
    parameters.push_back(param);
    log_prob.push_back(-2.0);       // expected log_prob

    param[0] = 0.25;                // y
    param[1] = 2.9;                 // alpha
    param[2] = 1.8;                 // sigma
    parameters.push_back(param);
    log_prob.push_back(-3.277093769205283724233);  // expected log_prob

    param[0] = 3.9;                 // y
    param[1] = 1.7;                 // alpha
    param[2] = 0.25;                // sigma
    parameters.push_back(param);
    log_prob.push_back(-102.8962074392704266756);  // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
    // alpha
    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // sigma
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);
  }

  template <typename T_y, typename T_shape, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_shape, T_scale>::type 
  log_prob(const T_y& y, const T_shape& alpha, const T_scale& sigma,
           const T3&, const T4&, const T5&) {
    return stan::math::weibull_log(y, alpha, sigma);
  }

  template <bool propto, 
            typename T_y, typename T_shape, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_shape, T_scale>::type 
  log_prob(const T_y& y, const T_shape& alpha, const T_scale& sigma,
           const T3&, const T4&, const T5&) {
    return stan::math::weibull_log<propto>(y, alpha, sigma);
  }
  
  
  template <typename T_y, typename T_shape, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_shape, T_scale>::type 
  log_prob_function(const T_y& y, const T_shape& alpha, const T_scale& sigma,
                    const T3&, const T4&, const T5&) {
    using std::log;
    using std::pow;
    using stan::math::multiply_log;

    return log(alpha) + multiply_log(alpha-1.0, y) 
      - multiply_log(alpha, sigma) - pow(y / sigma, alpha);
  }
};

TEST(ProbDistributionsWeibull,Cumulative) {
  using stan::math::weibull_cdf;
  using std::numeric_limits;
  EXPECT_FLOAT_EQ(0.86466472, weibull_cdf(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.0032585711, weibull_cdf(0.25,2.9,1.8));
  EXPECT_FLOAT_EQ(1.0, weibull_cdf(3.9,1.7,0.25));

  // EXPECT_FLOAT_EQ(0.0,
  //                 weibull_cdf(-numeric_limits<double>::infinity(),
  //                             1.0,1.0));
  EXPECT_FLOAT_EQ(0.0, weibull_cdf(0.0,1.0,1.0));
  EXPECT_FLOAT_EQ(1.0, weibull_cdf(numeric_limits<double>::infinity(),
                                   1.0,1.0));
}
