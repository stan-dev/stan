#define _LOG_PROB_ beta_log
#include <stan/prob/distributions/univariate/continuous/beta.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;

class ProbDistributionsBeta : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 0.2;           // y
    param[1] = 1.0;           // alpha
    param[2] = 1.0;           // beta
    parameters.push_back(param);
    log_prob.push_back(0.0); // expected log_prob

    param[0] = 0.3;           // y
    param[1] = 12.0;          // alpha
    param[2] = 25.0;          // beta
    parameters.push_back(param);
    log_prob.push_back(1.628758); // expected log_prob
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

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsBeta,
			      DistributionTestFixture,
			      ProbDistributionsBeta);



TEST(ProbDistributionsBeta,Cumulative) {
  EXPECT_FLOAT_EQ(0.06590419, stan::prob::beta_cdf(0.3,0.7,0.1));
  EXPECT_FLOAT_EQ(0.1565002, stan::prob::beta_cdf(0.7,0.7,0.1));
  EXPECT_FLOAT_EQ(0.5, stan::prob::beta_cdf(0.5,5,5));
  EXPECT_FLOAT_EQ(0.2665677, stan::prob::beta_cdf(0.4,5,5));
  EXPECT_FLOAT_EQ(0.8828741, stan::prob::beta_cdf(0.66,2,3));
  EXPECT_FLOAT_EQ(0.8734666, stan::prob::beta_cdf(0.66,2.1,3));
}
TEST(ProbDistributionsBeta,CumulativeDefaultPolicySigma) {
  EXPECT_THROW(stan::prob::beta_cdf(0.5, 0.0, 1), std::domain_error)
    << "alpha == 0 should throw";
  EXPECT_THROW(stan::prob::beta_cdf(0.5, -0.5, 1), std::domain_error)
    << "alpha < 0 should throw";
  EXPECT_THROW(stan::prob::beta_cdf(0.5, 2, 0), std::domain_error)
    << "beta == 0 should throw";
  EXPECT_THROW(stan::prob::beta_cdf(0.5, 2, -1), std::domain_error)
    << "beta < 0 should throw";
}
