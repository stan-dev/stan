#include <gtest/gtest.h>
#include <stan/prob/distributions/multivariate/discrete/categorical.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributionsCategorical,Categorical) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  EXPECT_FLOAT_EQ(-1.203973, stan::prob::categorical_log(0,theta));
  EXPECT_FLOAT_EQ(-0.6931472, stan::prob::categorical_log(1,theta));
}
TEST(ProbDistributionsCategorical,Propto) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  EXPECT_FLOAT_EQ(0.0, stan::prob::categorical_log<true>(0,theta));
  EXPECT_FLOAT_EQ(0.0, stan::prob::categorical_log<true>(1,theta));
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

using stan::prob::categorical_log;

TEST(ProbDistributionsCategorical,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  unsigned int n = 1;
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;

  EXPECT_NO_THROW(categorical_log(n, theta));
  EXPECT_NO_THROW(categorical_log(0, theta));
  EXPECT_NO_THROW(categorical_log(2, theta));
  
 
  EXPECT_THROW(categorical_log(3, theta), std::domain_error);
  theta(0) = nan;
  EXPECT_THROW(categorical_log(n, theta), std::domain_error);
  theta(0) = inf;
  EXPECT_THROW(categorical_log(n, theta), std::domain_error);
  theta(0) = -inf;
  EXPECT_THROW(categorical_log(n, theta), std::domain_error);
  theta(0) = -1;
  theta(1) = 1;
  theta(2) = 0;
  EXPECT_THROW(categorical_log(n, theta), std::domain_error);
}
TEST(ProbDistributionsCategorical,ErrnoPolicy) {  
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  double result;
  unsigned int n = 1;
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;

  result = categorical_log(n, theta, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = categorical_log(0, theta, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = categorical_log(2, theta, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  
 
  result = categorical_log(3, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  theta(0) = nan;
  result = categorical_log(n, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  theta(0) = inf;
  result = categorical_log(n, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  theta(0) = -inf;
  result = categorical_log(n, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  theta(0) = -1;
  theta(1) = 1;
  theta(2) = 0;
  result = categorical_log(n, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
