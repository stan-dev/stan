#define _LOG_PROB_ binomial_log
#include <stan/prob/distributions/univariate/discrete/binomial.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_2_discrete_1_param.hpp>

using std::vector;
using std::log;
using std::numeric_limits;

class ProbDistributionsBinomial : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 10;           // n
    param[1] = 20;           // N
    param[2] = 0.4;          // theta
    parameters.push_back(param);
    log_prob.push_back(-2.144372); // expected log_prob

    param[0] = 5;            // n
    param[1] = 15;           // N
    param[2] = 0.8;          // theta
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
    
    // theta
    index.push_back(2U);
    value.push_back(-1e-15);
    
    index.push_back(2U);
    value.push_back(1.0+1e-15);

  }
};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsBinomial,
			      DistributionTestFixture,
			      ProbDistributionsBinomial);

