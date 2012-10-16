#define _LOG_PROB_ weibull_log
#include <stan/prob/distributions/univariate/continuous/weibull.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;

class ProbDistributionsWeibull : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 2.0;                 // y
    param[1] = 1.0;                 // alpha
    param[2] = 1.0;                 // sigma
    parameters.push_back(param);
    log_prob.push_back(-2.0);       // expected log_prob

    param[0] = 0.25;                // y
    param[1] = 2.9;                 // alpha
    param[2] = 1.8;                 // sigma
    parameters.push_back(param);
    log_prob.push_back(-3.277094);  // expected log_prob

    param[0] = 3.9;                 // y
    param[1] = 1.7;                 // alpha
    param[2] = 0.25;                // sigma
    parameters.push_back(param);
    log_prob.push_back(-102.8962);  // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // y
    
    // alpha
    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // sigma
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsWeibull,
			      DistributionTestFixture,
			      ProbDistributionsWeibull);

TEST(ProbDistributionsWeibull,Cumulative) {
  using stan::prob::weibull_cdf;
  using std::numeric_limits;
  EXPECT_FLOAT_EQ(0.86466472, weibull_cdf(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.0032585711, weibull_cdf(0.25,2.9,1.8));
  EXPECT_FLOAT_EQ(1.0, weibull_cdf(3.9,1.7,0.25));

  // ??
  EXPECT_FLOAT_EQ(0.0,
                  weibull_cdf(-numeric_limits<double>::infinity(),
                              1.0,1.0));
  EXPECT_FLOAT_EQ(0.0, weibull_cdf(0.0,1.0,1.0));
  EXPECT_FLOAT_EQ(1.0, weibull_cdf(numeric_limits<double>::infinity(),
                                   1.0,1.0));
}
