// These tests should not have reference to stan::agrad::var. Distribution tests 
// with stan::agrad::var should be placed in src/test/agrad/distributions_test.cpp

#include <gtest/gtest.h>
#include "stan/prob/distributions_inv_wishart.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributions,InvWishart) {
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
