#include <gtest/gtest.h>
#include "stan/prob/distributions/multi_normal.hpp"

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


TEST(ProbDistributionsMultiNormal,MultiNormal) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_log(y,mu,Sigma));
}
TEST(ProbDistributionsMultiNormal,DefaultPolicySigma) {
  Matrix<double,Dynamic,1> y(2,1);
  y << 2.0, -2.0;
  Matrix<double,Dynamic,1> mu(2,1);
  mu << 1.0, -1.0;
  Matrix<double,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 9.0, -3.0, -3.0, 4.0;
  EXPECT_NO_THROW (stan::prob::multi_normal_log(y, mu, Sigma));

  // non-symmetric
  Sigma(0, 1) = -2.5;
  EXPECT_THROW (stan::prob::multi_normal_log(y, mu, Sigma), std::domain_error);
}
TEST(ProbDistributionsMultiNormal,ErrnoPolicySigma) {
  Matrix<double,Dynamic,1> y(2,1);
  y << 2.0, -2.0;
  Matrix<double,Dynamic,1> mu(2,1);
  mu << 1.0, -1.0;
  Matrix<double,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 9.0, -3.0, -3.0, 4.0;
  
  double result;
  EXPECT_NO_THROW (result=stan::prob::multi_normal_log(y, mu, Sigma, errno_policy()));

  // non-symmetric
  Sigma(0, 1) = -2.5;
  EXPECT_NO_THROW (result=stan::prob::multi_normal_log(y, mu, Sigma, errno_policy()));
  EXPECT_TRUE(std::isnan(result)) << "non-symmetric Sigma should return nan.";
}


