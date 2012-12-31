#define _LOG_PROB_ uniform_log
#include <stan/prob/distributions/univariate/continuous/uniform.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;

class ProbDistributionsUniform : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 0.2;                 // y
    param[1] = 0.1;                 // alpha
    param[2] = 1.0;                 // beta
    parameters.push_back(param);
    log_prob.push_back(log(1/0.9));   // expected log_prob

    param[0] = 0.2;                 // y
    param[1] = -0.25;               // alpha
    param[2] = 0.25;                // beta
    parameters.push_back(param);
    log_prob.push_back(log(2.0));   // expected log_prob

    param[0] = 101;                 // y
    param[1] = 100;                 // alpha
    param[2] = 110;                 // beta
    parameters.push_back(param);
    log_prob.push_back(log(0.1));   // expected log_prob
  }
 
  void invalid_values(vector<size_t>& /*index*/, 
                      vector<double>& /*value*/) {
    // y
    
    // alpha

    // beta
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsUniform,
                              DistributionTestFixture,
                              ProbDistributionsUniform);
