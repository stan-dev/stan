#define _LOG_PROB_ cauchy_log
#include <stan/prob/distributions/univariate/continuous/cauchy.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>
using std::vector;
using std::numeric_limits;

class ProbDistributionsCauchy : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(3);
    
    param[0] = 1.0;                // y
    param[1] = 0.0;                // mu
    param[2] = 1.0;                // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.837877); // expected log_prob

    param[0] = -1.5;                // y
    param[1] = 0.0;                 // mu
    param[2] = 1.0;                 // sigma
    parameters.push_back(param);
    log_prob.push_back(-2.323385); // expected log_prob

    param[0] = -1.5;                // y
    param[1] = -1.0;                // mu
    param[2] = 1.0;                 // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.367873); // expected log_prob
  }

  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // y
    
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

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsCauchy,
			      DistributionTestFixture,
			      ProbDistributionsCauchy);

TEST(ProbDistributionsCauchy,Cumulative) {
  using stan::prob::cauchy_cdf;
  EXPECT_FLOAT_EQ(0.75, cauchy_cdf(1.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.187167, cauchy_cdf(-1.5, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.187167, cauchy_cdf(-2.5, -1.0, 1.0));
}
