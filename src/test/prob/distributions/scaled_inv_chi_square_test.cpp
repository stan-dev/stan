#include <gtest/gtest.h>
#include "stan/prob/distributions/scaled_inv_chi_square.hpp"

TEST(ProbDistributions,ScaledInvChiSquare) {
  EXPECT_FLOAT_EQ(-3.091965, stan::prob::scaled_inv_chi_square_log(12.7,6.1,3.0));
  EXPECT_FLOAT_EQ(-1.737086, stan::prob::scaled_inv_chi_square_log(1.0,1.0,0.5));
}
