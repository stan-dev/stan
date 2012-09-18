#define _LOG_PROB_ beta_log
#include <stan/prob/distributions/univariate/continuous/beta.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsBeta : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(3);

    param[0] = 0.2;           // y
    param[1] = 1.0;           // alpha
    param[2] = 1.0;           // beta
    parameters.push_back(param);

    param[0] = 0.3;           // y
    param[1] = 12.0;          // alpha
    param[2] = 25.0;          // beta
    parameters.push_back(param);
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

  template <class T_y, class T_scale1, class T_scale2>
  var log_prob(const T_y& y, const T_scale1& alpha, const T_scale2& beta) {
    using std::log;
    using stan::math::log1m;
    using stan::math::log;
    using stan::math::value_of;
    using stan::prob::include_summand;
    
    var logp(0);
    
    if (include_summand<true,T_y,T_scale1>::value)
      logp += (alpha - 1.0) * log(y);
    if (include_summand<true,T_y,T_scale2>::value)
      logp += (beta - 1.0) * log1m(y);
    if (include_summand<true,T_scale1,T_scale2>::value)
      logp += lgamma(alpha + beta);
    if (include_summand<true,T_scale1>::value)
      logp -= lgamma(alpha);
    if (include_summand<true,T_scale2>::value)
      logp -= lgamma(beta);
    return logp;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsBeta,
			      AgradDistributionTestFixture,
			      AgradDistributionsBeta);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsBeta,
			      AgradDistributionTestFixture2,
			      AgradDistributionsBeta);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsBeta,
			      AgradDistributionTestFixture3,
			      AgradDistributionsBeta);
