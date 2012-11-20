#define _LOG_PROB_ lognormal_log
#include <stan/prob/distributions/univariate/continuous/lognormal.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsLognormal : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(3);

    param[0] = 1.2;           // y
    param[1] = 0.3;           // mu
    param[2] = 1.5;           // sigma
    parameters.push_back(param);

    param[0] = 12.0;          // y
    param[1] = 0.3;           // mu
    param[2] = 1.5;           // sigma
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
  
  template <class T_y, class T_loc, class T_scale,
	    typename T3, typename T4, typename T5, 
	    typename T6, typename T7, typename T8, typename T9>
  var log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
	       const T3&, const T4&, const T5&, 
	       const T6&, const T7&, const T8&, const T9&) {
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

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsLognormal,
			      AgradDistributionTestFixture,
			      AgradDistributionsLognormal);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsLognormal,
			      AgradDistributionTestFixture2,
			      AgradDistributionsLognormal);
