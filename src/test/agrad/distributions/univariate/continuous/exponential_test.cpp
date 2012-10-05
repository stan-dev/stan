#define _LOG_PROB_ exponential_log
#include <stan/prob/distributions/univariate/continuous/exponential.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_2_params.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsExponential : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(2);

    param[0] = 2.0;                 // y
    param[1] = 1.5;                 // beta
    parameters.push_back(param);

    param[0] = 15.0;                // y
    param[1] = 3.9;                 // beta
    parameters.push_back(param);
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

  template <class T_y, class T_inv_scale>
  var log_prob(const T_y& y, const T_inv_scale& beta) {
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

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsExponential,
			      AgradDistributionTestFixture,
			      AgradDistributionsExponential);
