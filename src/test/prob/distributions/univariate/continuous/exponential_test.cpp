#define _LOG_PROB_ exponential_log
#include <stan/prob/distributions/univariate/continuous/exponential.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_2_params.hpp>

using std::vector;
using std::numeric_limits;

class ProbDistributionsExponential : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(2);

    param[0] = 2.0;                 // y
    param[1] = 1.5;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-2.594535);  // expected log_prob

    param[0] = 15.0;                // y
    param[1] = 3.9;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-57.13902);  // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // y
    
    // beta
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsExponential,
			      DistributionTestFixture,
			      ProbDistributionsExponential);

TEST(ProbDistributionsExponential,Cumulative) {
  using std::numeric_limits;
  using stan::prob::exponential_cdf;
  EXPECT_FLOAT_EQ(0.95021293, exponential_cdf(2.0,1.5));
  EXPECT_FLOAT_EQ(1.0, exponential_cdf(15.0,3.9));
  EXPECT_FLOAT_EQ(0.62280765, exponential_cdf(0.25,3.9));

  // ??
  // EXPECT_FLOAT_EQ(0.0, 
  //                 exponential_cdf(-numeric_limits<double>::infinity(),
  //                                 1.5));
  EXPECT_FLOAT_EQ(0.0, exponential_cdf(0.0,1.5));
  EXPECT_FLOAT_EQ(1.0, 
                  exponential_cdf(numeric_limits<double>::infinity(),
                                  1.5));

}
