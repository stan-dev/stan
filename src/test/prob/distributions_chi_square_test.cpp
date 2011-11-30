// These tests should not have reference to stan::agrad::var. Distribution tests 
// with stan::agrad::var should be placed in src/test/agrad/distributions_test.cpp

#include <cmath>
#include <gtest/gtest.h>
#include "stan/prob/distributions_chi_square.hpp"

TEST(distributions,ChiSquare) {
  EXPECT_FLOAT_EQ(-3.835507, stan::prob::chi_square_log(7.9,3.0));
  EXPECT_FLOAT_EQ(-2.8927, stan::prob::chi_square_log(1.9,0.5));
}
TEST(distributions,ChiSquareDefaultPolicy) {
  double y = 0.0;
  double nu = 0.0;
  EXPECT_THROW(stan::prob::chi_square_log(y, nu), std::domain_error);
  EXPECT_THROW(stan::prob::chi_square_log(y, -1), std::domain_error);
  EXPECT_THROW(stan::prob::chi_square_log(-1, nu), std::domain_error);
}
