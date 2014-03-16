#include <gtest/gtest.h>
#include "stan/prob/distributions/multivariate/continuous/multi_normal_prec.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;
using std::vector;

TEST(ProbDistributionsMultiNormalPrec, MultiNormal) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<double,Dynamic,Dynamic> L = Sigma.inverse();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_prec_log(y,mu,L));
}

TEST(ProbDistributionsMultiNormalPrec,MultiNormal2) {
  vector< Matrix<double,Dynamic,1> > vec_y(2);
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  vec_y[0] = y;
  y << 4.0, -2.0, 1.0;
  vec_y[1] = y;
  vector< Matrix<double,Dynamic,1> > vec_mu(2);
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  vec_mu[0] = mu;
  mu << 2.0, -1.0, 4.0;
  vec_mu[1] = mu;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 10.0, -3.0, 0.0,
    -3.0,  5.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<double,Dynamic,Dynamic> L = Sigma.inverse();
  EXPECT_FLOAT_EQ(-11.928077-6.5378327, stan::prob::multi_normal_prec_log(vec_y,vec_mu,L));
}
TEST(ProbDistributionsMultiNormalPrec,MultiNormalVar) {
  using stan::agrad::var;
  Matrix<var,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<var,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<var,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<var,Dynamic,Dynamic> L = Sigma.inverse();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_prec_log(y,mu,L).val());
}
/*
TEST(ProbDistributionsMultiNormalPrec,MultiNormalOneRow) {
  Matrix<double,Dynamic,Dynamic> y(1,3);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<double,Dynamic,Dynamic> L = Sigma.inverse();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_prec_log(y,mu,L));
}

TEST(ProbDistributionsMultiNormalPrec,MultiNormalMultiRow) {
  Matrix<double,Dynamic,Dynamic> y(2,3);
  y << 2.0, -2.0, 11.0,
       4.0, -4.0, 22.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<double,Dynamic,Dynamic> L = Sigma.inverse();
  EXPECT_FLOAT_EQ(-54.2152, stan::prob::multi_normal_prec_log(y,mu,L));
}
*/
