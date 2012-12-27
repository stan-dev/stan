#define _LOG_PROB_ beta_binomial_log
#include <stan/prob/distributions/univariate/discrete/beta_binomial.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_2_discrete_2_params.hpp>

using std::vector;
using std::log;
using std::numeric_limits;

class ProbDistributionsBetaBinomial : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(4);


    param[0] = 5;            // n
    param[1] = 20;           // N
    param[2] = 10.0;         // alpha
    param[3] = 25.0;         // beta
    parameters.push_back(param);
    log_prob.push_back(-1.854007); // expected log_prob

    param[0] = 25;           // n
    param[1] = 100;          // N
    param[2] = 30.0;         // alpha
    param[3] = 50.0;         // beta
    parameters.push_back(param);
    log_prob.push_back(-4.376696); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // n
    
    // N
    index.push_back(1U);
    value.push_back(-1);
    
    // alpha
    index.push_back(2U);
    value.push_back(0.0);
    
    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(std::numeric_limits<double>::infinity());

    // beta
    index.push_back(3U);
    value.push_back(0.0);
    
    index.push_back(3U);
    value.push_back(-1.0);

    index.push_back(3U);
    value.push_back(std::numeric_limits<double>::infinity());
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsBetaBinomial,
			      DistributionTestFixture,
			      ProbDistributionsBetaBinomial);

TEST(ProbDistributionsBetaBinomialCDF,Values) {
    EXPECT_FLOAT_EQ(0.8868204314, stan::prob::beta_binomial_cdf(49, 100, 1.349, 3.938));
}

// #include <gtest/gtest.h>
// #include <stan/prob/distributions/univariate/discrete/beta_binomial.hpp>

// TEST(ProbDistributionsBetaBinomial,DefaultPolicy) {
//   double nan = std::numeric_limits<double>::quiet_NaN();
//   double inf = std::numeric_limits<double>::infinity();

//   int n = 5;
//   int N = 15;
//   double alpha = 3.0;
//   double beta = 4.5;

// }
// TEST(ProbDistributionsBetaBinomial,ErrnoPolicy) {
//   double nan = std::numeric_limits<double>::quiet_NaN();
//   double inf = std::numeric_limits<double>::infinity();

//   double result;
//   int n = 5;
//   int N = 15;
//   double alpha = 3.0;
//   double beta = 4.5;

//   result = beta_binomial_log(n,N,alpha,beta, errno_policy());
//   EXPECT_FALSE(std::isnan(result));
//   result = beta_binomial_log(n,0,alpha,beta, errno_policy());
//   EXPECT_FALSE(std::isnan(result));
  
//   result = beta_binomial_log(n,-1,alpha,beta, errno_policy());
//   EXPECT_TRUE(std::isnan(result));
  
//   result = beta_binomial_log(n,N,nan,beta, errno_policy());
//   EXPECT_TRUE(std::isnan(result));
//   result = beta_binomial_log(n,N,0.0,beta, errno_policy());
//   EXPECT_TRUE(std::isnan(result));
//   result = beta_binomial_log(n,N,-1.0,beta, errno_policy());
//   EXPECT_TRUE(std::isnan(result));
//   result = beta_binomial_log(n,N,-inf,beta, errno_policy());
//   EXPECT_TRUE(std::isnan(result));
//   result = beta_binomial_log(n,N,inf,beta, errno_policy());
//   EXPECT_TRUE(std::isnan(result));
  
//   result = beta_binomial_log(n,N,alpha,nan, errno_policy());
//   EXPECT_TRUE(std::isnan(result));
//   result = beta_binomial_log(n,N,alpha,0.0, errno_policy());
//   EXPECT_TRUE(std::isnan(result));
//   result = beta_binomial_log(n,N,alpha,-1.0, errno_policy());
//   EXPECT_TRUE(std::isnan(result));
//   result = beta_binomial_log(n,N,alpha,-inf, errno_policy());
//   EXPECT_TRUE(std::isnan(result));
//   result = beta_binomial_log(n,N,alpha,inf, errno_policy());
//   EXPECT_TRUE(std::isnan(result));
// }
