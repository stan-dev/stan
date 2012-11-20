#define _LOG_PROB_ inv_gamma_log
#include <stan/prob/distributions/univariate/continuous/inv_gamma.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsInvGamma : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(3);

    param[0] = 1.0;                 // y
    param[1] = 1.0;                 // alpha
    param[2] = 1.0;                 // beta
    parameters.push_back(param);

    param[0] = 0.5;                 // y
    param[1] = 2.9;                 // alpha
    param[2] = 3.1;                 // beta
    parameters.push_back(param);
    
    /*
      param[0] = 0.0;                 // y
      param[1] = 2.9;                 // alpha
      param[2] = 3.1;                 // beta
      parameters.push_back(param);
    */
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

  template <class T_y, class T_shape, class T_scale,
	    typename T3, typename T4, typename T5, 
	    typename T6, typename T7, typename T8, typename T9>
  var log_prob(const T_y& y, const T_shape& alpha, const T_scale& beta,
	       const T3&, const T4&, const T5&, 
	       const T6&, const T7&, const T8&, const T9&) {
    if (y <= 0)
      return stan::prob::LOG_ZERO;
    
    using boost::math::lgamma;
    using stan::math::multiply_log;
    using stan::prob::include_summand;
    
    var lp = 0.0;
    if (include_summand<true,T_shape>::value)
      lp -= lgamma(alpha);
    if (include_summand<true,T_shape,T_scale>::value)
      lp += multiply_log(alpha,beta);
    if (include_summand<true,T_y,T_shape>::value)
      lp -= multiply_log(alpha+1.0, y);
    if (include_summand<true,T_y,T_scale>::value)
      lp -= beta / y;
    return lp;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsInvGamma,
			      AgradDistributionTestFixture,
			      AgradDistributionsInvGamma);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsInvGamma,
			      AgradDistributionTestFixture2,
			      AgradDistributionsInvGamma);
