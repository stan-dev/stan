#include <gtest/gtest.h>
#include "stan/prob/distributions/multivariate/continuous/multi_normal.hpp"
#include "stan/prob/distributions/multivariate/continuous/multi_gp.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;

using boost::math::policies::policy;
using boost::math::policies::evaluation_error;
using boost::math::policies::domain_error;
using boost::math::policies::overflow_error;
using boost::math::policies::domain_error;
using boost::math::policies::pole_error;
using boost::math::policies::errno_on_error;

// NOTE: what does this do?
typedef policy<
  domain_error<errno_on_error>, 
  pole_error<errno_on_error>,
  overflow_error<errno_on_error>,
  evaluation_error<errno_on_error> 
  > errno_policy;

TEST(ProbDistributionsMultiGP,LoglikeUU) {
  Matrix<double, 1, 1> FF;
  Matrix<double, 1, 1> GG;
  Matrix<double, 1, 1> V;
  Matrix<double, 1, 1> W;

  FF << 0.585528817843856;
  GG << -0.109303314681054;
  V << 2.25500747900521;
  W << 0.461487989960454;

  Matrix<double,Dynamic,1> y(10);
  y << -191.283411318882, 19.2723905416, -1.43621233172613, -1.59717406907683, 
    0.66578474366854, 0.21545711451668, -1.3710009374748, -3.11525241456936, 
    -0.193395632057508, 0.714053334632461;

  double expected = 13.1327101955358;

  lp_ref = gaussian_dlm_log(y, FF, GG, V, W);

  EXPECT_FLOAT_EQ(lp_ref, expected);
}

// TEST(ProbDistributionsMultiGP,DefaultPolicySigma) {
//   Matrix<double,Dynamic,Dynamic> y(3,5);
//   y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;
  
//   Matrix<double,Dynamic,Dynamic> Sigma(5,5);
//   Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
//   -3.0,  4.0, 0.0,  0.0, 0.0,
//   0.0,  0.0, 5.0,  1.0, 0.0,
//   0.0,  0.0, 1.0, 10.0, 0.0,
//   0.0,  0.0, 0.0,  0.0, 2.0;
  
//   Matrix<double,Dynamic,1> w(3,1);
//   w << 1.0, 0.5, 1.5;
  
//   // non-symmetric
//   Sigma(0, 1) = -2.5;
//   EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
//   Sigma(0, 1) = Sigma(1, 0);

//   // non-spd
//   Sigma(0, 0) = -3.0;
//   EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
//   Sigma(0, 1) = 9.0;
// }

// TEST(ProbDistributionsMultiGP,DefaultPolicyW) {
//   Matrix<double,Dynamic,Dynamic> y(3,5);
//   y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;
  
//   Matrix<double,Dynamic,Dynamic> Sigma(5,5);
//   Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
//   -3.0,  4.0, 0.0,  0.0, 0.0,
//   0.0,  0.0, 5.0,  1.0, 0.0,
//   0.0,  0.0, 1.0, 10.0, 0.0,
//   0.0,  0.0, 0.0,  0.0, 2.0;
  
//   Matrix<double,Dynamic,1> w(3,1);
//   w << 1.0, 0.5, 1.5;
  
//   // negative w
//   w(0, 0) = -2.5;
//   EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);

//   // non-finite values
//   w(0, 0) = std::numeric_limits<double>::infinity();
//   EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
//   w(0, 0) = -std::numeric_limits<double>::infinity();
//   EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
//   w(0,0) = std::numeric_limits<double>::quiet_NaN();
//   EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
// }

// TEST(ProbDistributionsMultiGP,DefaultPolicyY) {
//   Matrix<double,Dynamic,Dynamic> y(3,5);
//   y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;
  
//   Matrix<double,Dynamic,Dynamic> Sigma(5,5);
//   Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
//   -3.0,  4.0, 0.0,  0.0, 0.0,
//   0.0,  0.0, 5.0,  1.0, 0.0,
//   0.0,  0.0, 1.0, 10.0, 0.0,
//   0.0,  0.0, 0.0,  0.0, 2.0;
  
//   Matrix<double,Dynamic,1> w(3,1);
//   w << 1.0, 0.5, 1.5;
  
//   // non-finite values
//   y(0, 0) = std::numeric_limits<double>::infinity();
//   EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
//   y(0, 0) = -std::numeric_limits<double>::infinity();
//   EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
//   y(0,0) = std::numeric_limits<double>::quiet_NaN();
//   EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
// }

