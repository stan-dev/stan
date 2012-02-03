#include <gtest/gtest.h>
#include <stan/prob/distributions/multivariate/discrete/multinomial.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(ProbDistributions,Multinomial) {
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  EXPECT_FLOAT_EQ(-2.002481, stan::prob::multinomial_log(ns,theta));
}
TEST(ProbDistributions,MultinomialPropto) {
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  EXPECT_FLOAT_EQ(0.0, stan::prob::multinomial_log<true>(ns,theta));
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

using stan::prob::multinomial_log;

TEST(ProbDistributionsMultinomial,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  
  EXPECT_NO_THROW(multinomial_log(ns, theta));
  
  ns[1] = 0;
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
  ns[1] = -1;
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
  ns[1] = 1;

  theta(0) = 0.0;
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
  theta(0) = nan;
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
  theta(0) = inf;
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
  theta(0) = -inf;
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
  theta(0) = -1;
  theta(1) = 1.5;
  theta(2) = 0.5;
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
  theta(0) = 0.2;
  theta(1) = 0.3;
  theta(2) = 0.5;
  
  ns.resize(2);
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
}

TEST(ProbDistributionsMultinomial,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  double result;
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  
  result = multinomial_log(ns, theta, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  
  ns[1] = 0;
  result = multinomial_log(ns, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  ns[1] = -1;
  result = multinomial_log(ns, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  ns[1] = 1;

  theta(0) = 0.0;
  result = multinomial_log(ns, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  theta(0) = nan;
  result = multinomial_log(ns, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  theta(0) = inf;
  result = multinomial_log(ns, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  theta(0) = -inf;
  result = multinomial_log(ns, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  theta(0) = -1;
  theta(1) = 1.5;
  theta(2) = 0.5;
  result = multinomial_log(ns, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  theta(0) = 0.2;
  theta(1) = 0.3;
  theta(2) = 0.5;
  
  ns.resize(2);
  result = multinomial_log(ns, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
