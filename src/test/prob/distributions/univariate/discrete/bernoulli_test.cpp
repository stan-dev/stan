#define _LOG_PROB_ bernoulli_log
#include <stan/prob/distributions/univariate/discrete/bernoulli.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/discrete_distribution_tests_2_params.hpp>

using std::vector;
using std::log;
using std::numeric_limits;

class ProbDistributionsBernoulli : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(2);

    param[0] = 1;           // n
    param[1] = 0.25;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.25)); // expected log_prob

    param[0] = 0;           // n
    param[1] = 0.25;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.75)); // expected log_prob

    param[0] = 1;           // n
    param[1] = 0.01;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.01)); // expected log_prob

    param[0] = 0;           // n
    param[1] = 0.01;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.99)); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1);

    index.push_back(0U);
    value.push_back(2);

    // theta
    index.push_back(1U);
    value.push_back(-0.001);

    index.push_back(1U);
    value.push_back(1.001);
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsBernoulli,
			      DistributionTestFixture,
			      ProbDistributionsBernoulli);


/*



#include <gtest/gtest.h>
#include "stan/prob/distributions/univariate/discrete/bernoulli.hpp"
#include "stan/math/special_functions.hpp"

TEST(ProbDistributionsBernoulli,Bernoulli) {
  EXPECT_FLOAT_EQ(std::log(0.25), stan::prob::bernoulli_log(1,0.25));
  EXPECT_FLOAT_EQ(std::log(1.0 - 0.25), stan::prob::bernoulli_log(0,0.25));
}
TEST(ProbDistributionsBernoulli,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::bernoulli_log<true>(1,0.25));
  EXPECT_FLOAT_EQ(0.0, stan::prob::bernoulli_log<true>(0,0.25));
}

using boost::math::policies::policy;
using boost::math::policies::evaluation_error;
using boost::math::policies::domain_error;
using boost::math::policies::overflow_error;
using boost::math::policies::domain_error;
using boost::math::policies::pole_error;
using boost::math::policies::errno_on_error;

typedef policy<
  domain_error<errno_on_error>, 
  pole_error<errno_on_error>,
  overflow_error<errno_on_error>,
  evaluation_error<errno_on_error> 
  > errno_policy;

using stan::prob::bernoulli_log;

TEST(ProbDistributionsBernoulli,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  unsigned int k = 1;
  double theta = 0.75;

  EXPECT_NO_THROW(bernoulli_log(k, theta));
  EXPECT_NO_THROW(bernoulli_log(k, 0.0));
  EXPECT_NO_THROW(bernoulli_log(k, 1.0));
    
  EXPECT_THROW(bernoulli_log(2U, theta), std::domain_error);
  EXPECT_THROW(bernoulli_log(k, nan), std::domain_error);
  EXPECT_THROW(bernoulli_log(k, inf), std::domain_error);
  EXPECT_THROW(bernoulli_log(k, -inf), std::domain_error);
  EXPECT_THROW(bernoulli_log(k, -1.0), std::domain_error);
  EXPECT_THROW(bernoulli_log(k, 2.0), std::domain_error);
}
TEST(ProbDistributionsBernoulli,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  double result;
  unsigned int k = 1;
  double theta = 0.75;

  result = bernoulli_log(k, theta, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = bernoulli_log(k, 0.0, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = bernoulli_log(k, 1.0, errno_policy());
  EXPECT_FALSE(std::isnan(result));
    
  result = bernoulli_log(2U, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = bernoulli_log(k, nan, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = bernoulli_log(k, inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = bernoulli_log(k, -inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = bernoulli_log(k, -1.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = bernoulli_log(k, 2.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}

// Same tests as above, but for bernoulli_logit.
const static double logit_025 = stan::math::logit(0.25);
TEST(ProbDistributionsBernoulliLogit,Bernoulli) {
  EXPECT_FLOAT_EQ(std::log(0.25), stan::prob::bernoulli_logit_log(1,logit_025));
  EXPECT_FLOAT_EQ(std::log(1.0 - 0.25), stan::prob::bernoulli_logit_log(0,logit_025));
  EXPECT_FLOAT_EQ(-std::exp(-25), stan::prob::bernoulli_logit_log(1, 25));
  EXPECT_FLOAT_EQ(-25, stan::prob::bernoulli_logit_log(0, 25));
  EXPECT_FLOAT_EQ(-25, stan::prob::bernoulli_logit_log(1, -25));
  EXPECT_FLOAT_EQ(-std::exp(-25), stan::prob::bernoulli_logit_log(0, -25));
}
TEST(ProbDistributionsBernoulliLogit,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::bernoulli_logit_log<true>(1,logit_025));
  EXPECT_FLOAT_EQ(0.0, stan::prob::bernoulli_logit_log<true>(0,logit_025));
}

typedef policy<
  domain_error<errno_on_error>, 
  pole_error<errno_on_error>,
  overflow_error<errno_on_error>,
  evaluation_error<errno_on_error> 
  > errno_policy;

using stan::prob::bernoulli_logit_log;

TEST(ProbDistributionsBernoulliLogit,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  unsigned int k = 1;
  double theta = 0.75;

  EXPECT_NO_THROW(bernoulli_logit_log(k, theta));
  EXPECT_NO_THROW(bernoulli_logit_log(k, 0.0));
  EXPECT_NO_THROW(bernoulli_logit_log(k, 1.0));
    
  EXPECT_THROW(bernoulli_logit_log(2U, theta), std::domain_error);
  EXPECT_THROW(bernoulli_logit_log(k, nan), std::domain_error);
  EXPECT_NO_THROW(bernoulli_logit_log(k, inf));
  EXPECT_NO_THROW(bernoulli_logit_log(k, -inf));
  EXPECT_NO_THROW(bernoulli_logit_log(k, -1.0));
  EXPECT_NO_THROW(bernoulli_logit_log(k, 2.0));
}
TEST(ProbDistributionsBernoulliLogit,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  double result;
  unsigned int k = 1;
  double theta = 0.75;

  result = bernoulli_logit_log(k, theta, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = bernoulli_logit_log(k, -inf, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = bernoulli_logit_log(k, inf, errno_policy());
  EXPECT_FALSE(std::isnan(result));
    
  result = bernoulli_logit_log(2U, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = bernoulli_logit_log(k, nan, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = bernoulli_logit_log(k, inf, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = bernoulli_logit_log(k, -inf, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = bernoulli_logit_log(k, -1.0, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = bernoulli_logit_log(k, 2.0, errno_policy());
  EXPECT_FALSE(std::isnan(result));
}
*/
