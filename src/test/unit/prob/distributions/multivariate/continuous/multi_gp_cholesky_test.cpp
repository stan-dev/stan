#include <gtest/gtest.h>
#include "stan/prob/distributions/multivariate/continuous/multi_normal.hpp"
#include "stan/prob/distributions/multivariate/continuous/multi_gp_cholesky.hpp"

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


TEST(ProbDistributionsMultiGPCholesky,MultiGPCholesky) {
  Matrix<double,Dynamic,1> mu(5,1);
  mu.setZero();
  
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();

  Matrix<double,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  double lp_ref(0);
  for (size_t i = 0; i < 3; i++) {
    Matrix<double,Dynamic,1> cy(y.row(i).transpose());
    Matrix<double,Dynamic,Dynamic> cSigma((1.0/w[i])*Sigma);
    lp_ref += stan::prob::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref, stan::prob::multi_gp_cholesky_log(y,L,w));
}

TEST(ProbDistributionsMultiGPCholesky,DefaultPolicyL) {
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();

  // TODO
}

TEST(ProbDistributionsMultiGPCholesky,DefaultPolicyW) {
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();

  Matrix<double,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  // negative w
  w(0, 0) = -2.5;
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);

  // non-finite values
  w(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);
  w(0, 0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);
  w(0,0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);
}

TEST(ProbDistributionsMultiGPCholesky,DefaultPolicyY) {
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();

  Matrix<double,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  // non-finite values
  y(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);
  y(0, 0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);
  y(0,0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);
}

