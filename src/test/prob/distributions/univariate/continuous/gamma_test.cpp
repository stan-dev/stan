#define _LOG_PROB_ gamma_log
#include <stan/prob/distributions/univariate/continuous/gamma.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;

class ProbDistributionsGamma : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.0;                 // y
    param[1] = 2.0;                 // alpha
    param[2] = 2.0;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-0.6137056); // expected log_prob

    param[0] = 2.0;                 // y
    param[1] = 0.25;                // alpha
    param[2] = 0.75;                // beta
    parameters.push_back(param);
    log_prob.push_back(-3.379803);  // expected log_prob

    param[0] = 1.0;                 // y
    param[1] = 1.0;                 // alpha
    param[2] = 1.0;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-1.0);       // expected log_prob
    
    /*param[0] = 0.0;                 // y
    param[1] = 1.0;                 // alpha
    param[2] = 2.0;                 // beta
    parameters.push_back(param);
    log_prob.push_back(log(2.0));   // expected log_prob
    
    param[0] = -10.0;               // y
    param[1] = 1.0;                 // alpha
    param[2] = 2.0;                 // beta
    parameters.push_back(param);
    log_prob.push_back(log(0.0));   // expected log_prob */
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // y
    
    // alpha
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // beta
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

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsGamma,
			      DistributionTestFixture,
			      ProbDistributionsGamma);

/*
// FIXME: include when gamma_cdf works.
TEST(ProbDistributionsGamma,Cumulative) {
  // values from R
  EXPECT_FLOAT_EQ(0.59399415, stan::prob::gamma_cdf(1.0,2.0,2.0));
  EXPECT_FLOAT_EQ(0.96658356, stan::prob::gamma_cdf(2.0,0.25,0.75));
  EXPECT_FLOAT_EQ(0.63212056, stan::prob::gamma_cdf(1,1,1));
  EXPECT_FLOAT_EQ(0.0, stan::prob::gamma_cdf(0,1,1));
}
*/
