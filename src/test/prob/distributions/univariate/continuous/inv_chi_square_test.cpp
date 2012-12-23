#define _LOG_PROB_ inv_chi_square_log
#include <stan/prob/distributions/univariate/continuous/inv_chi_square.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_2_params.hpp>

using std::vector;
using std::numeric_limits;

class ProbDistributionsInvChiSquare : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(2);

    param[0] = 0.5;                 // y
    param[1] = 2.0;                 // nu
    parameters.push_back(param);
    log_prob.push_back(-0.3068528);  // expected log_prob

    param[0] = 3.2;                 // y
    param[1] = 9.1;                 // nu
    parameters.push_back(param);
    log_prob.push_back(-12.28905);    // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // y
    
    // nu
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsInvChiSquare,
			      DistributionTestFixture,
			      ProbDistributionsInvChiSquare);

TEST(ProbDistributionsInvChiSquareCdf,Values) {
    EXPECT_FLOAT_EQ(0.067889154, stan::prob::inv_chi_square_cdf(0.3, 1.0));
}