#define _LOG_PROB_ logistic_log
#include <stan/prob/distributions/univariate/continuous/logistic.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsLogistic : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(3);

    param[0] = 1.2;           // y
    param[1] = 0.3;           // mu
    param[2] = 2.0;           // sigma
    parameters.push_back(param);

    param[0] = -1.0;          // y
    param[1] = 0.2;           // mu
    param[2] = 0.25;          // sigma
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
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());
  }

  template <class T_y, class T_loc, class T_scale>
  var log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      using stan::prob::include_summand;
      using stan::math::log1p;
      var lp(0.0);
      if (include_summand<true,T_y,T_loc,T_scale>::value)
	lp -= (y - mu) / sigma;
      if (include_summand<true,T_scale>::value)
	lp -= log(sigma);
      if (include_summand<true,T_y,T_loc,T_scale>::value)
	lp -= 2.0 * log1p(exp(-(y - mu)/sigma));
      return lp;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsLogistic,
                              AgradDistributionTestFixture,
                              AgradDistributionsLogistic);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsLogistic,
                              AgradDistributionTestFixture2,
                              AgradDistributionsLogistic);
