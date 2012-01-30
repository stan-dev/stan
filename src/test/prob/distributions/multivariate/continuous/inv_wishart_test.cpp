#include <gtest/gtest.h>
#include <stan/prob/distributions/multivariate/continuous/inv_wishart.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

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

using stan::prob::inv_wishart_log;



TEST(ProbDistributionsInvWishart,InvWishart) {
  Matrix<double,Dynamic,Dynamic> Y(3,3);
  Y <<  12.147233, -11.9036079, 1.0910458,
    -11.903608,  16.7585782, 0.8530256,
    1.091046,   0.8530256, 2.5786609;

  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 7.785215,  3.0597878,  1.1071663,
    3.059788, 10.3515035, -0.1232598,
    1.107166, -0.1232598,  7.7623386;
  
  double dof = 4.0;
  double log_p = log(2.008407e-08);

  EXPECT_NEAR(log_p, stan::prob::inv_wishart_log(Y,dof,Sigma), 0.01);
}
TEST(ProbDistributionsInvWishart,Propto) {
  Matrix<double,Dynamic,Dynamic> Y(3,3);
  Y <<  12.147233, -11.9036079, 1.0910458,
    -11.903608,  16.7585782, 0.8530256,
    1.091046,   0.8530256, 2.5786609;

  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 7.785215,  3.0597878,  1.1071663,
    3.059788, 10.3515035, -0.1232598,
    1.107166, -0.1232598,  7.7623386;
  
  double dof = 4.0;
  
  EXPECT_FLOAT_EQ(0.0, stan::prob::inv_wishart_log<true>(Y,dof,Sigma));
}
TEST(ProbDistributionsInvWishart,DefaultPolicy) {
  Matrix<double,Dynamic,Dynamic> Sigma;
  Matrix<double,Dynamic,Dynamic> Y;
  double nu;
  
  Sigma.resize(1,1);
  Y.resize(1,1);
  nu = 1;
  EXPECT_NO_THROW(inv_wishart_log(Y, nu, Sigma));
  
  nu = 5;
  Sigma.resize(2,1);
  EXPECT_THROW(inv_wishart_log(Y, nu, Sigma), std::domain_error);

  nu = 5;
  Sigma.resize(2,2);
  Y.resize(2,1);
  EXPECT_THROW(inv_wishart_log(Y, nu, Sigma), std::domain_error);
  
  nu = 5;
  Sigma.resize(2,2);
  Y.resize(3,3);
  EXPECT_THROW(inv_wishart_log(Y, nu, Sigma), std::domain_error);

  Sigma.resize(3,3);
  Y.resize(3,3);
  nu = 2;
  EXPECT_NO_THROW(inv_wishart_log(Y, nu, Sigma));
  nu = 1;
  EXPECT_THROW(inv_wishart_log(Y, nu, Sigma), std::domain_error);
}
TEST(ProbDistributionsInvWishart,ErrnoPolicy) {
  Matrix<double,Dynamic,Dynamic> Sigma;
  Matrix<double,Dynamic,Dynamic> Y;
  double nu;
  double result;
  
  Sigma.resize(1,1);
  Y.resize(1,1);
  Sigma << 1;
  Y << 1;
  nu = 1;
  result = inv_wishart_log(Y, nu, Sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));

  nu = 5;
  Sigma.resize(2,1);
  result = inv_wishart_log(Y, nu, Sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));

  nu = 5;
  Sigma.resize(2,2);
  Y.resize(2,1);
  result = inv_wishart_log(Y, nu, Sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  nu = 5;
  Sigma.resize(2,2);
  Y.resize(3,3);
  result = inv_wishart_log(Y, nu, Sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));

  Sigma.resize(3,3);
  Sigma << 1,0,0, 0,1,0, 0,0,1;
  Y.resize(3,3);
  Y << 1,0,0, 0,1,0, 0,0,1;
  nu = 2;
  result = inv_wishart_log(Y, nu, Sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  
  nu = 1;
  result = inv_wishart_log(Y, nu, Sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
