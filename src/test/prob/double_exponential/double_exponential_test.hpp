// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/double_exponential_log.hpp>
#include <stan/math/prim/scal/prob/double_exponential_cdf.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsDoubleExponential : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);
    
    param[0] = 1.0;                  // y
    param[1] = 1.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.6931471805599453094172321214581765680755001343602552541206800094933936219696947156058633269964186875);  // expected log_prob

    param[0] = 2.0;                  // y
    param[1] = 1.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.693147180559945309417232121458176568075500134360255254120680009493393621969694715605863326996418688);   // expected log_prob
    
    param[0] = -3.0;                 // y
    param[1] = 2.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-5.693147180559945309417232121458176568075500134360255254120680009493393621969694715605863326996418688);   // expected log_prob
    
    param[0] = 1.0;                  // y
    param[1] = 0.0;                  // mu
    param[2] = 2.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.8862943611198906188344642429163531361510002687205105082413600189867872439393894312117266539928373751);   // expected log_prob

    param[0] = 1.9;                  // y
    param[1] = 2.3;                  // mu
    param[2] = 0.5;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.8);        // expected log_prob

    param[0] = 1.9;                  // y
    param[1] = 2.3;                  // mu
    param[2] = 0.25;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.90685281944005469058276788);        // expected log_prob
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
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
           const T3&, const T4&, const T5&) {
    return stan::math::double_exponential_log(y, mu, sigma);
  }

  template <bool propto, 
            typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
           const T3&, const T4&, const T5&) {
    return stan::math::double_exponential_log<propto>(y, mu, sigma);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
                        const T3&, const T4&, const T5&) {
    using std::log;
    using std::fabs;
    using stan::math::NEG_LOG_TWO;

    return NEG_LOG_TWO- log(sigma) - fabs(y - mu) / sigma;
  }
};


TEST(ProbDistributionsDoubleExponential,Cumulative) {
  EXPECT_FLOAT_EQ(0.5, stan::math::double_exponential_cdf(1.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.8160603, stan::math::double_exponential_cdf(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.003368973, stan::math::double_exponential_cdf(-3.0,2.0,1.0));
  EXPECT_FLOAT_EQ(0.6967347, stan::math::double_exponential_cdf(1.0,0.0,2.0));
  EXPECT_FLOAT_EQ(0.2246645, stan::math::double_exponential_cdf(1.9,2.3,0.5));
  EXPECT_FLOAT_EQ(0.10094826, stan::math::double_exponential_cdf(1.9,2.3,0.25));
}

