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
TEST(ProbDistributions,GammaPropto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::gamma_log<true>(1.0,2.0,2.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::gamma_log<true>(2.0,0.25,0.75));
  EXPECT_FLOAT_EQ(0.0, stan::prob::gamma_log<true>(1,1,1));
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

TEST(ProbDistributionsCumulative,Gamma) {
  // values from R
  EXPECT_FLOAT_EQ(0.59399415, stan::prob::gamma_p(1.0,2.0,2.0));
  EXPECT_FLOAT_EQ(0.96658356, stan::prob::gamma_p(2.0,0.25,0.75));
  EXPECT_FLOAT_EQ(0.63212056, stan::prob::gamma_p(1,1,1));
  EXPECT_FLOAT_EQ(0.0, stan::prob::gamma_p(0,1,1));
}
