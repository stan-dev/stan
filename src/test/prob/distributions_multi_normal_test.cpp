#include <cmath>
#include <gtest/gtest.h>
#include <boost/exception/all.hpp>
#include <Eigen/Dense>

#include "stan/prob/distributions_multi_normal.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;


TEST(ProbDistributions,MultiNormal) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_log<double>(y,mu,Sigma));
}
TEST(ProbDistributions,MultiNormalDefaultPolicy) {
  Matrix<double,Dynamic,1> y(2,1);
  y << 2.0, -2.0;
  Matrix<double,Dynamic,1> mu(2,1);
  mu << 1.0, -1.0;
  Matrix<double,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 9.0, -3.0, -3.0, 4.0;
  EXPECT_NO_THROW (stan::prob::multi_normal_log<double>(y, mu, Sigma));

  // non-symmetric
  Sigma(0, 1) = -2.5;
  EXPECT_THROW (stan::prob::multi_normal_log<double>(y, mu, Sigma), std::domain_error);
  Sigma(0, 1) = Sigma(1,0);
}

