// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/inv_gamma_log.hpp>
#include <stan/math/prim/scal/prob/inv_gamma_cdf.hpp>

#include <stan/math/prim/scal/fun/multiply_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsInvGamma : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.0;                 // y
    param[1] = 1.0;                 // alpha
    param[2] = 1.0;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-1.0);       // expected log_prob

    param[0] = 0.5;                 // y
    param[1] = 2.9;                 // alpha
    param[2] = 3.1;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-0.8185294827413338580868); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
    // alpha
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

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
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_shape, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_shape, T_scale>::type 
  log_prob(const T_y& y, const T_shape& alpha, const T_scale& beta,
           const T3&, const T4&, const T5&) {
    return stan::math::inv_gamma_log(y, alpha, beta);
  }

  template <bool propto, 
            typename T_y, typename T_shape, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_shape, T_scale>::type 
  log_prob(const T_y& y, const T_shape& alpha, const T_scale& beta,
           const T3&, const T4&, const T5&) {
    return stan::math::inv_gamma_log<propto>(y, alpha, beta);
  }
  
  
  template <typename T_y, typename T_shape, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_shape, T_scale>::type 
  log_prob_function(const T_y& y, const T_shape& alpha, const T_scale& beta,
                    const T3&, const T4&, const T5&) {
    if (y <= 0)
      return stan::math::LOG_ZERO;
    
    using boost::math::lgamma;
    using stan::math::multiply_log;

    return -lgamma(alpha) + multiply_log(alpha,beta) 
      - multiply_log(alpha+1.0, y) - beta / y;
  }
};

TEST(ProbDistributionsInvGammaCdf,Values) {
    EXPECT_FLOAT_EQ(0.557873, stan::math::inv_gamma_cdf(4.39, 1.349, 3.938));
}
