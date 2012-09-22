#define _LOG_PROB_ bernoulli_logit_log
#include <stan/prob/distributions/univariate/discrete/bernoulli.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/discrete_distribution_tests_2_params.hpp>

using std::vector;
using std::log;
using std::numeric_limits;

class ProbDistributionsBernoulliLogit : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    using stan::math::logit;
    using std::exp;
    vector<double> param(2);

    param[0] = 1;           // n
    param[1] = logit(0.25); // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.25)); // expected log_prob

    param[0] = 0;           // n
    param[1] = logit(0.25); // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.75)); // expected log_prob

    param[0] = 1;           // n
    param[1] = logit(0.01); // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.01)); // expected log_prob

    param[0] = 0;           // n
    param[1] = logit(0.01); // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.99)); // expected log_prob

    param[0] = 0;            // n
    param[1] = 25;           // theta
    parameters.push_back(param);
    log_prob.push_back(-25); // expected log_prob

    param[0] = 1;            // n
    param[1] = -25;          // theta
    parameters.push_back(param);
    log_prob.push_back(-25); // expected log_prob
    
    param[0] = 0;           // n
    param[1] = -25;         // theta
    parameters.push_back(param);
    log_prob.push_back(-exp(-25)); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1);

    index.push_back(0U);
    value.push_back(2);

    // theta
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsBernoulliLogit,
			      DistributionTestFixture,
			      ProbDistributionsBernoulliLogit);

