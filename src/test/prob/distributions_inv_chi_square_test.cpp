// These tests should not have reference to stan::agrad::var. Distribution tests 
// with stan::agrad::var should be placed in src/test/agrad/distributions_test.cpp

#include <cmath>
#include <gtest/gtest.h>
#include "stan/prob/distributions_inv_chi_square.hpp"

TEST(ProbDistributions,InvChiSquare) {
  EXPECT_FLOAT_EQ(-0.3068528, stan::prob::inv_chi_square_log(0.5,2.0));
  EXPECT_FLOAT_EQ(-12.28905, stan::prob::inv_chi_square_log(3.2,9.1));
}
