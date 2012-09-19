#define _LOG_PROB_ cauchy_log
#include <stan/prob/distributions/univariate/continuous/cauchy.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsCauchy : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(3);

    param[0] = 1.0;                // y
    param[1] = 0.0;                // mu
    param[2] = 1.0;                // sigma
    parameters.push_back(param);

    param[0] = -1.5;                // y
    param[1] = 0.0;                 // mu
    param[2] = 1.0;                 // sigma
    parameters.push_back(param);

    param[0] = -1.5;                // y
    param[1] = -1.0;                // mu
    param[2] = 1.0;                 // sigma
    parameters.push_back(param);
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

  template <typename T_y, typename T_loc, typename T_scale>
  var log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma) {
    using stan::math::log1p;
    using stan::math::square;
    using stan::prob::include_summand;
    
    var lp = 0.0;
    if (include_summand<true>::value)
      lp += stan::prob::NEG_LOG_PI;
    if (include_summand<true,T_scale>::value)
      lp -= log(sigma);
    if (include_summand<true,T_y,T_loc,T_scale>::value)
      lp -= log1p(square((y - mu) / sigma));
    return lp;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsCauchy,
			      AgradDistributionTestFixture,
			      AgradDistributionsCauchy);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsCauchy,
			      AgradDistributionTestFixture2,
			      AgradDistributionsCauchy);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsCauchy,
			      AgradDistributionTestFixture3,
			      AgradDistributionsCauchy);
