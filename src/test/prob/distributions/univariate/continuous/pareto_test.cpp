#define _LOG_PROB_ pareto_log
#include <stan/prob/distributions/univariate/continuous/pareto.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;

class ProbDistributionsPareto : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.5;           // y
    param[1] = 0.5;           // y_min
    param[2] = 2.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-1.909543); // expected log_prob

    param[0] = 19.5;          // y
    param[1] = 0.15;          // y_min
    param[2] = 5.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-25.69865); // expected log_prob

    param[0] = 0.0;           // y
    param[1] = 0.15;          // y_min
    param[2] = 5.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(log(0.0)); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    
    // y_min
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // alpha
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsPareto,
                              DistributionTestFixture,
                              ProbDistributionsPareto);

TEST(ProbDistributionsParetoCDF, Values) {
    EXPECT_FLOAT_EQ(0.60434447, stan::prob::pareto_cdf(3.45, 2.89, 5.235));
}
