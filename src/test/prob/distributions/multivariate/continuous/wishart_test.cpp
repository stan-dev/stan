#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <stan/prob/distributions/univariate/continuous/chi_square.hpp>
#include <stan/prob/distributions/multivariate/continuous/wishart.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributionsWishart,2x2) {
  Matrix<double,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 1.848220, 1.899623, 
    1.899623, 12.751941;

  Matrix<double,Dynamic,Dynamic> Y(2,2);
  Y <<  2.011108, -11.20661,
    -11.206611, 112.94139;

  unsigned int dof = 3;
  
  double lp = log(8.658e-07); // computed with MCMCpack in R
 
  EXPECT_NEAR(lp, stan::prob::wishart_log(Y,dof,Sigma), 0.01);
}
TEST(ProbDistributionsWishart,4x4) {
  Matrix<double,Dynamic,Dynamic> Y(4,4);
  Y << 7.988168,  -9.555605, -14.47483,   4.395895,
    -9.555605,  44.750570,  49.21577, -18.454186,
    -14.474830,  49.215769,  60.08987, -21.481079,
    4.395895, -18.454186, -21.48108, 7.885833;
  
  Matrix<double,Dynamic,Dynamic> Sigma(4,4);
  Sigma << 2.9983662,  0.2898776, -2.650523,  0.1055911,
    0.2898776, 11.4803610,  7.157993, -3.1129955,
    -2.6505229,  7.1579931, 11.676181, -3.5866852,
    0.1055911, -3.1129955, -3.586685,  1.4482736;

  double dof = 4;
  double log_p = log(8.034197e-10);
  EXPECT_NEAR(log_p, stan::prob::wishart_log(Y,dof,Sigma),0.01);
  
  dof = 5;
  log_p = log(1.517951e-10);
  EXPECT_NEAR(log_p, stan::prob::wishart_log(Y,dof,Sigma),0.01);
}
TEST(ProbDistributionsWishart,2x2Propto) {
  Matrix<double,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 1.848220, 1.899623, 
    1.899623, 12.751941;

  Matrix<double,Dynamic,Dynamic> Y(2,2);
  Y <<  2.011108, -11.20661,
    -11.206611, 112.94139;

  unsigned int dof = 3;
 
  EXPECT_FLOAT_EQ(0.0, stan::prob::wishart_log<true>(Y,dof,Sigma));
}
TEST(ProbDistributionsWishart,4x4Propto) {
  Matrix<double,Dynamic,Dynamic> Y(4,4);
  Y << 7.988168,  -9.555605, -14.47483,   4.395895,
    -9.555605,  44.750570,  49.21577, -18.454186,
    -14.474830,  49.215769,  60.08987, -21.481079,
    4.395895, -18.454186, -21.48108, 7.885833;
  
  Matrix<double,Dynamic,Dynamic> Sigma(4,4);
  Sigma << 2.9983662,  0.2898776, -2.650523,  0.1055911,
    0.2898776, 11.4803610,  7.157993, -3.1129955,
    -2.6505229,  7.1579931, 11.676181, -3.5866852,
    0.1055911, -3.1129955, -3.586685,  1.4482736;

  double dof = 4;
  EXPECT_FLOAT_EQ(0.0, stan::prob::wishart_log<true>(Y,dof,Sigma));
  
  dof = 5;
  EXPECT_FLOAT_EQ(0.0, stan::prob::wishart_log<true>(Y,dof,Sigma));
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

using stan::prob::wishart_log;

TEST(ProbDistributionsWishart,DefaultPolicy) {
  Matrix<double,Dynamic,Dynamic> Sigma;
  Matrix<double,Dynamic,Dynamic> Y;
  double nu;
  
  Sigma.resize(1,1);
  Y.resize(1,1);
  nu = 1;
  EXPECT_NO_THROW(wishart_log(Y, nu, Sigma));
  
  nu = 5;
  Sigma.resize(2,1);
  EXPECT_THROW(wishart_log(Y, nu, Sigma), std::domain_error);

  nu = 5;
  Sigma.resize(2,2);
  Y.resize(2,1);
  EXPECT_THROW(wishart_log(Y, nu, Sigma), std::domain_error);
  
  nu = 5;
  Sigma.resize(2,2);
  Y.resize(3,3);
  EXPECT_THROW(wishart_log(Y, nu, Sigma), std::domain_error);

  Sigma.resize(3,3);
  Y.resize(3,3);
  nu = 2;
  EXPECT_NO_THROW(wishart_log(Y, nu, Sigma));
  nu = 1;
  EXPECT_THROW(wishart_log(Y, nu, Sigma), std::domain_error);
}
TEST(ProbDistributionsWishart,ErrnoPolicy) {
  Matrix<double,Dynamic,Dynamic> Sigma;
  Matrix<double,Dynamic,Dynamic> Y;
  double nu;
  double result;
  
  Sigma.resize(1,1);
  Y.resize(1,1);
  Sigma << 1;
  Y << 1;
  nu = 1;
  result = wishart_log(Y, nu, Sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));

  nu = 5;
  Sigma.resize(2,1);
  result = wishart_log(Y, nu, Sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));

  nu = 5;
  Sigma.resize(2,2);
  Y.resize(2,1);
  result = wishart_log(Y, nu, Sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  nu = 5;
  Sigma.resize(2,2);
  Y.resize(3,3);
  result = wishart_log(Y, nu, Sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));

  Sigma.resize(3,3);
  Sigma << 1,0,0, 0,1,0, 0,0,1;
  Y.resize(3,3);
  Y << 1,0,0, 0,1,0, 0,0,1;
  nu = 2;
  result = wishart_log(Y, nu, Sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  
  nu = 1;
  result = wishart_log(Y, nu, Sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}

TEST(ProbDistroWishart,chiSqEquiv) {
  double s = 1.9;
  double w = 2.5;
  double nu = 2.0;

  Matrix<double,Dynamic,Dynamic> Sigma(1,1);
  Sigma(0,0) = s;
  Matrix<double,Dynamic,Dynamic> W(1,1);
  W(0,0) = w;

  

  EXPECT_FLOAT_EQ(stan::prob::chi_square_log<false>(w, nu/2.0, 1.0/(2.0 * w)),
                  stan::prob::wishart_log(W,nu,Sigma));
}
