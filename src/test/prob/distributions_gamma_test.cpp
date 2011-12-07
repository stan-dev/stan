// These tests should not have reference to stan::agrad::var. Distribution tests 
// with stan::agrad::var should be placed in src/test/agrad/distributions_test.cpp

#include <gtest/gtest.h>
#include "stan/prob/distributions_gamma.hpp"

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributions,Gamma) {
  EXPECT_FLOAT_EQ(-0.6137056, stan::prob::gamma_log(1.0,2.0,2.0));
  EXPECT_FLOAT_EQ(-3.379803, stan::prob::gamma_log(2.0,0.25,0.75));
  EXPECT_FLOAT_EQ(-1, stan::prob::gamma_log(1,1,1));
}
TEST(ProbDistributions,GammaDefaultPolicy) {
  double y = 0;
  double alpha = 1.0;
  double beta = 2.0;
  
  EXPECT_NO_THROW(stan::prob::gamma_log(y, alpha, beta));
  EXPECT_THROW (stan::prob::gamma_log(-1, alpha, beta), std::domain_error);
  EXPECT_THROW (stan::prob::gamma_log(y, 0.0, beta), std::domain_error);
  EXPECT_THROW (stan::prob::gamma_log(y, -1.0, beta), std::domain_error);
  EXPECT_THROW (stan::prob::gamma_log(y, alpha, 0.0), std::domain_error);
  EXPECT_THROW (stan::prob::gamma_log(y, alpha, -1.0), std::domain_error);
}
