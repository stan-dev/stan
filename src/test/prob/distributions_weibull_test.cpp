// These tests should not have reference to stan::agrad::var. Distribution tests 
// with stan::agrad::var should be placed in src/test/agrad/distributions_test.cpp

#include <gtest/gtest.h>
#include "stan/prob/distributions.hpp"

TEST(prob_prob,weibull) {
  EXPECT_FLOAT_EQ(-2.0, stan::prob::weibull_log(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(-3.277094, stan::prob::weibull_log(0.25,2.9,1.8));
  EXPECT_FLOAT_EQ(-102.8962, stan::prob::weibull_log(3.9,1.7,0.25));
}

