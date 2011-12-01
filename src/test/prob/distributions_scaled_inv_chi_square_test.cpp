// These tests should not have reference to stan::agrad::var. Distribution tests 
// with stan::agrad::var should be placed in src/test/agrad/distributions_test.cpp

#include <cmath>
#include <gtest/gtest.h>
#include "stan/prob/distributions_scaled_inv_chi_square.hpp"

TEST(distributions,ScaledInvChiSquare) {
  EXPECT_FLOAT_EQ(-3.091965, stan::prob::scaled_inv_chi_square_log(12.7,6.1,3.0));
  EXPECT_FLOAT_EQ(-1.737086, stan::prob::scaled_inv_chi_square_log(1.0,1.0,0.5));
}
