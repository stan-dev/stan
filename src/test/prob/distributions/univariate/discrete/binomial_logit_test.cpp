#define _LOG_PROB_ binomial_logit_log
#include <stan/prob/distributions/univariate/discrete/binomial.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_2_discrete_1_param.hpp>

using std::vector;
using std::log;
using std::numeric_limits;

class ProbDistributionsBinomialLogit : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    using stan::math::logit;

    param[0] = 10;           // n
    param[1] = 20;           // N
    param[2] = logit(0.4);          // alpha
    parameters.push_back(param);
    log_prob.push_back(-2.144372); // expected log_prob

    param[0] = 5;            // n
    param[1] = 15;           // N
    param[2] = logit(0.8);          // alpha
    parameters.push_back(param);
    log_prob.push_back(-9.20273); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);
    
    
    // N
    index.push_back(1U);
    value.push_back(-1);
    
    // alpha
    index.push_back(2U);
    value.push_back(std::numeric_limits<double>::infinity());

  }
};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsBinomialLogit,
                              DistributionTestFixture,
                              ProbDistributionsBinomialLogit);
