#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/matrix_normal_prec_log.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributionsMatrixNormal,MatrixNormalPrec) {
  Matrix<double,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
       11.0, 2.0, -5.0, 11.0, 0.0,
       -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<double,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
       0.5, 1.0, 0.2,
       0.1, 0.2, 1.0;
  
  double lp_ref;
  lp_ref = stan::math::matrix_normal_prec_log(y,mu,D,Sigma);
  EXPECT_FLOAT_EQ(lp_ref,-2132.0748232368409845);
}

TEST(ProbDistributionsMatrixNormal, ErrorSigma) {
  Matrix<double,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
  11.0, 2.0, -5.0, 11.0, 0.0,
  -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
  0.5, 1.0, 0.2,
  0.1, 0.2, 1.0;
  
  // non-symmetric
  Sigma(0, 1) = -2.5;
  EXPECT_THROW (stan::math::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
  Sigma(0, 1) = Sigma(1, 0);

  // non-spd
  Sigma(0, 0) = -3.0;
  EXPECT_THROW (stan::math::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
  Sigma(0, 0) = 9.0;
}

TEST(ProbDistributionsMatrixNormal, ErrorD) {
  Matrix<double,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
  11.0, 2.0, -5.0, 11.0, 0.0,
  -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
  0.5, 1.0, 0.2,
  0.1, 0.2, 1.0;
  
  // non-symmetric
  D(0, 1) = -2.5;
  EXPECT_THROW (stan::math::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
  D(0, 1) = Sigma(1, 0);
  
  // non-spd
  D(0, 0) = -3.0;
  EXPECT_THROW (stan::math::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
  D(0, 0) = 1.0;
}

TEST(ProbDistributionsMatrixNormal, ErrorY) {
  Matrix<double,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
  11.0, 2.0, -5.0, 11.0, 0.0,
  -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
  0.5, 1.0, 0.2,
  0.1, 0.2, 1.0;
  
  // non-finite values
  y(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::math::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
  y(0, 0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::math::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
  y(0,0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::math::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
}
