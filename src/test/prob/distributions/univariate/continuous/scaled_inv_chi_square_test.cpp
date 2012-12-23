#define _LOG_PROB_ scaled_inv_chi_square_log
#include <stan/prob/distributions/univariate/continuous/scaled_inv_chi_square.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;

class ProbDistributionsScaledChiSquare : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 12.7;          // y
    param[1] = 6.1;           // nu
    param[2] = 3.0;           // s
    parameters.push_back(param);
    log_prob.push_back(-3.091965); // expected log_prob

    param[0] = 1.0;           // y
    param[1] = 1.0;           // nu
    param[2] = 0.5;           // s
    parameters.push_back(param);
    log_prob.push_back(-1.737086); // expected log_prob
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

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // s
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsScaledChiSquare,
			      DistributionTestFixture,
			      ProbDistributionsScaledChiSquare);

TEST(ProbDistributionsScaledInvChiSquareCDF, Values) {
    EXPECT_FLOAT_EQ(0.37242326, stan::prob::scaled_inv_chi_square_cdf(4.39, 1.349, 1.984));
}