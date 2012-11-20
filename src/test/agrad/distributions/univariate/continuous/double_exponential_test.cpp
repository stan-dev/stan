#define _LOG_PROB_ double_exponential_log
#include <stan/prob/distributions/univariate/continuous/double_exponential.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsDoubleExponential : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(3);
	       
    param[0] = 1.0;                  // y
    param[1] = 1.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);

    param[0] = 2.0;                  // y
    param[1] = 1.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    
    param[0] = -3.0;                 // y
    param[1] = 2.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    
    param[0] = 1.0;                  // y
    param[1] = 0.0;                  // mu
    param[2] = 2.0;                  // sigma
    parameters.push_back(param);

    param[0] = 1.9;                  // y
    param[1] = 2.3;                  // mu
    param[2] = 0.5;                  // sigma
    parameters.push_back(param);

    param[0] = 1.9;                  // y
    param[1] = 2.3;                  // mu
    param[2] = 0.25;                  // sigma
    parameters.push_back(param);
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

  template <class T_y, class T_loc, class T_scale,
	    typename T3, typename T4, typename T5, 
	    typename T6, typename T7, typename T8, typename T9>
  var log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
	       const T3&, const T4&, const T5&, 
	       const T6&, const T7&, const T8&, const T9&) {
    using std::log;
    using stan::math::log;
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

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsDoubleExponential,
			      AgradDistributionTestFixture,
			      AgradDistributionsDoubleExponential);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsDoubleExponential,
			      AgradDistributionTestFixture2,
			      AgradDistributionsDoubleExponential);
