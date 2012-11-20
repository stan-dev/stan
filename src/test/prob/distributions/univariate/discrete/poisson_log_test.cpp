#define _LOG_PROB_ poisson_log_log
#include <stan/prob/distributions/univariate/discrete/poisson.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_1_discrete_1_param.hpp>

using std::vector;
using std::log;
using std::numeric_limits;

class ProbDistributionsPoissonLog : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {

    using std::log;

    vector<double> param(2);

    param[0] = 17;           // n
    param[1] = log(13.0);         // alpha
    parameters.push_back(param);
    log_prob.push_back(-2.900934); // expected log_prob

    param[0] = 192;          // n
    param[1] = log(42.0);         // alpha
    parameters.push_back(param);
    log_prob.push_back(-145.3547); // expected log_prob

    param[0] = 0;            // n
    param[1] = log(3.0);          // alpha
    parameters.push_back(param);
    log_prob.push_back(-3.0); // expected log_prob

    param[0] = 0;            // n
    param[1] = std::numeric_limits<double>::infinity(); // alpha
    parameters.push_back(param);
    log_prob.push_back(log(0.0)); // expected log_prob

    param[0] = 1;            // n
    param[1] = log(0.0);          // alpha
    parameters.push_back(param);
    log_prob.push_back(log(0.0)); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);

    // alpha
    // all OK
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsPoissonLog,
                              DistributionTestFixture,
                              ProbDistributionsPoissonLog);

