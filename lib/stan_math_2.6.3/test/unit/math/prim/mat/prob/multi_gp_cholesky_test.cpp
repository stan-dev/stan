#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/multi_gp_cholesky_log.hpp>
#include <stan/math/prim/mat/prob/multi_normal_log.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

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
    lp_ref += stan::math::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref, stan::math::multi_gp_cholesky_log(y,L,w));
}

TEST(ProbDistributionsMultiGPCholesky,ErrorL) {
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

TEST(ProbDistributionsMultiGPCholesky,ErrorW) {
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
  EXPECT_THROW (stan::math::multi_gp_cholesky_log(y, L, w), std::domain_error);

  // non-finite values
  w(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::math::multi_gp_cholesky_log(y, L, w), std::domain_error);
  w(0, 0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::math::multi_gp_cholesky_log(y, L, w), std::domain_error);
  w(0,0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::math::multi_gp_cholesky_log(y, L, w), std::domain_error);
}

TEST(ProbDistributionsMultiGPCholesky,ErrorY) {
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
  EXPECT_THROW (stan::math::multi_gp_cholesky_log(y, L, w), std::domain_error);
  y(0, 0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::math::multi_gp_cholesky_log(y, L, w), std::domain_error);
  y(0,0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::math::multi_gp_cholesky_log(y, L, w), std::domain_error);
}
