#define _LOG_PROB_ double_exponential_log
#include <stan/prob/distributions/univariate/continuous/double_exponential.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;

class ProbDistributionsDoubleExponential : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(3);
    
    param[0] = 1.0;                  // y
    param[1] = 1.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.6931472);  // expected log_prob

    param[0] = 2.0;                  // y
    param[1] = 1.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.693147);   // expected log_prob
    
    param[0] = -3.0;                 // y
    param[1] = 2.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-5.693147);   // expected log_prob
    
    param[0] = 1.0;                  // y
    param[1] = 0.0;                  // mu
    param[2] = 2.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.886294);   // expected log_prob

    param[0] = 1.9;                  // y
    param[1] = 2.3;                  // mu
    param[2] = 0.5;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.8);        // expected log_prob

    param[0] = 1.9;                  // y
    param[1] = 2.3;                  // mu
    param[2] = 0.25;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.9068528);        // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(0U);
    value.push_back(-numeric_limits<double>::infinity());
    
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
};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsDoubleExponential,
 			      DistributionTestFixture,
 			      ProbDistributionsDoubleExponential);

TEST(ProbDistributionsDoubleExponential,Cumulative) {
  EXPECT_FLOAT_EQ(0.5, stan::prob::double_exponential_cdf(1.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.8160603, stan::prob::double_exponential_cdf(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.003368973, stan::prob::double_exponential_cdf(-3.0,2.0,1.0));
  EXPECT_FLOAT_EQ(0.6967347, stan::prob::double_exponential_cdf(1.0,0.0,2.0));
  EXPECT_FLOAT_EQ(0.2246645, stan::prob::double_exponential_cdf(1.9,2.3,0.5));
  EXPECT_FLOAT_EQ(0.10094826, stan::prob::double_exponential_cdf(1.9,2.3,0.25));
}
