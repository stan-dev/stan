#define _LOG_PROB_ poisson_log
#include <stan/prob/distributions/univariate/discrete/poisson.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_1_discrete_1_param.hpp>

using std::vector;
using std::log;
using std::numeric_limits;

class ProbDistributionsPoisson : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(2);

    param[0] = 17;           // n
    param[1] = 13.0;         // lambda
    parameters.push_back(param);
    log_prob.push_back(-2.900934); // expected log_prob

    param[0] = 192;          // n
    param[1] = 42.0;         // lambda
    parameters.push_back(param);
    log_prob.push_back(-145.3547); // expected log_prob

    param[0] = 0;            // n
    param[1] = 3.0;          // lambda
    parameters.push_back(param);
    log_prob.push_back(-3.0); // expected log_prob

    param[0] = 0;            // n
    param[1] = std::numeric_limits<double>::infinity(); // lambda
    parameters.push_back(param);
    log_prob.push_back(log(0.0)); // expected log_prob

    param[0] = 1;            // n
    param[1] = 0.0;          // lambda
    parameters.push_back(param);
    log_prob.push_back(log(0.0)); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);

    // lambda
    index.push_back(1U);
    value.push_back(-1e-5);

    index.push_back(1U);
    value.push_back(-1);
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsPoisson,
			      DistributionTestFixture,
			      ProbDistributionsPoisson);

