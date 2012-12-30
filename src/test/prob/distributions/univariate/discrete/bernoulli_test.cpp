#define _LOG_PROB_ bernoulli_log
#include <stan/prob/distributions/univariate/discrete/bernoulli.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_1_discrete_1_param.hpp>

using std::vector;
using std::log;
using std::numeric_limits;

class ProbDistributionsBernoulli : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(2);

    param[0] = 1;           // n
    param[1] = 0.25;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.25)); // expected log_prob

    param[0] = 0;           // n
    param[1] = 0.25;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.75)); // expected log_prob

    param[0] = 1;           // n
    param[1] = 0.01;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.01)); // expected log_prob

    param[0] = 0;           // n
    param[1] = 0.01;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.99)); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1);

    index.push_back(0U);
    value.push_back(2);

    // theta
    index.push_back(1U);
    value.push_back(-0.001);

    index.push_back(1U);
    value.push_back(1.001);
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsBernoulli,
                              DistributionTestFixture,
                              ProbDistributionsBernoulli);

TEST(ProbDistributionsBernoulliCDF,Values) {
    EXPECT_FLOAT_EQ(1, stan::prob::bernoulli_cdf(1, 0.57));
    EXPECT_FLOAT_EQ(1 - 0.57, stan::prob::bernoulli_cdf(0, 0.57));
}
