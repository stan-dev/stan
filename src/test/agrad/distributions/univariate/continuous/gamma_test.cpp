#define _LOG_PROB_ gamma_log
#include <stan/prob/distributions/univariate/continuous/gamma.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsGamma : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(3);

    param[0] = 1.0;                 // y
    param[1] = 2.0;                 // alpha
    param[2] = 2.0;                 // beta
    parameters.push_back(param);

    param[0] = 2.0;                 // y
    param[1] = 0.25;                // alpha
    param[2] = 0.75;                // beta
    parameters.push_back(param);

    param[0] = 1.0;                 // y
    param[1] = 1.0;                 // alpha
    param[2] = 1.0;                 // beta
    parameters.push_back(param);
    
    /*
      // FIXME: add boundary values that won't be used in finite diff tests.
      param[0] = 0.0;                 // y
    param[1] = 1.0;                 // alpha
    param[2] = 2.0;                 // beta
    parameters.push_back(param);
    
    param[0] = -10.0;               // y
    param[1] = 1.0;                 // alpha
    param[2] = 2.0;                 // beta
    parameters.push_back(param);*/
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

  template <class T_y, class T_shape, class T_inv_scale,
	    typename T3, typename T4, typename T5, 
	    typename T6, typename T7, typename T8, typename T9>
  var log_prob(const T_y& y, const T_shape& alpha, const T_inv_scale& beta,
	       const T3&, const T4&, const T5&, 
	       const T6&, const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;
    using stan::math::multiply_log;
    
    var lp(0);
    if (include_summand<true,T_shape>::value)
      lp -= lgamma(alpha);
    if (include_summand<true,T_shape,T_inv_scale>::value)
      lp += multiply_log(alpha,beta);
    if (include_summand<true,T_y,T_shape>::value)
      lp += multiply_log(alpha-1.0,y);
    if (include_summand<true,T_y,T_inv_scale>::value)
      lp -= beta * y;
    return lp;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsGamma,
			      AgradDistributionTestFixture,
			      AgradDistributionsGamma);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsGamma,
			      AgradDistributionTestFixture2,
			      AgradDistributionsGamma);

/*
// FIXME: include once gamma_cdf works.
TEST(AgradDistributionsGamma,GammaCdf) {
  double y(1.0);
  var y_var(1.0);
  double alpha(1.0);
  var alpha_var(1.0);
  double beta(1.0);
  var beta_var(1.0);
  EXPECT_FLOAT_EQ(0.59399415, stan::prob::gamma_cdf(1.0,2.0,2.0));
  EXPECT_FLOAT_EQ(0.59399415, stan::prob::gamma_cdf(y_var,alpha_var,beta_var));
  }
*/
