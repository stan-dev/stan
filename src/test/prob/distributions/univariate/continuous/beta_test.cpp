#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/continuous/beta.hpp>

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

using stan::prob::beta_log;


TEST(ProbDistributionsBeta,Beta) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::beta_log(0.2,1.0,1.0));
  EXPECT_FLOAT_EQ(1.628758, stan::prob::beta_log(0.3,12.0,25.0));
  // EXPECT_FLOAT_EQ(0.0, stan::prob::beta_ls_log(0.2,0.5,2.0));
  // EXPECT_FLOAT_EQ(1.628758, stan::prob::beta_ls_log(0.3,12.0 / 37.0,37.0));
}
TEST(ProbDistributionsBeta,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::beta_log<true>(0.2,1.0,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::beta_log<true>(0.3,12.0,25.0));
  //  EXPECT_FLOAT_EQ(0.0, stan::prob::beta_ls_log<true>(0.2,0.5,2.0));
  // EXPECT_FLOAT_EQ(1.628758, stan::prob::beta_ls_log<true>(0.3,12.0 / 37.0,37.0));
}
TEST(ProbDistributionsBeta,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  double y = 0.5;
  double alpha = 2.0;
  double beta = 3.0;
  
  EXPECT_NO_THROW(beta_log(y, alpha, beta));
  EXPECT_NO_THROW(beta_log(inf, alpha, beta));
  EXPECT_NO_THROW(beta_log(-inf, alpha, beta));
  
  EXPECT_THROW(beta_log(nan, alpha, beta), std::domain_error);

  EXPECT_THROW(beta_log(y, 0.0, beta), std::domain_error);
  EXPECT_THROW(beta_log(y, -1.0, beta), std::domain_error);
  EXPECT_THROW(beta_log(y, -inf, beta), std::domain_error);
  EXPECT_THROW(beta_log(y, nan, beta), std::domain_error);
  EXPECT_THROW(beta_log(y, inf, beta), std::domain_error);

  EXPECT_THROW(beta_log(y, alpha, 0.0), std::domain_error);
  EXPECT_THROW(beta_log(y, alpha, -1.0), std::domain_error);
  EXPECT_THROW(beta_log(y, alpha, -inf), std::domain_error);
  EXPECT_THROW(beta_log(y, alpha, nan), std::domain_error);
  EXPECT_THROW(beta_log(y, alpha, inf), std::domain_error);
}
TEST(ProbDistributionsBeta,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  double result;
  double y = 0.5;
  double alpha = 2.0;
  double beta = 3.0;
  
  result = beta_log(y, alpha, beta, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = beta_log(inf, alpha, beta, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = beta_log(-inf, alpha, beta, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  
  result = beta_log(nan, alpha, beta, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));

  result = beta_log(y, 0.0, beta, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = beta_log(y, -1.0, beta, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = beta_log(y, -inf, beta, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = beta_log(y, nan, beta, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = beta_log(y, inf, beta, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));

  result = beta_log(y, alpha, 0.0, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = beta_log(y, alpha, -1.0, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = beta_log(y, alpha, -inf, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = beta_log(y, alpha, nan, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = beta_log(y, alpha, inf, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));

}
