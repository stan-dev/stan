// These tests should not have reference to stan::agrad::var. Distribution tests 
// with stan::agrad::var should be placed in src/test/agrad/distributions_test.cpp

#include <cmath>
#include <gtest/gtest.h>
#include "stan/prob/distributions_exponential.hpp"

TEST(distributions,Exponential) {
  EXPECT_FLOAT_EQ(-2.594535, stan::prob::exponential_log(2.0,1.5));
  EXPECT_FLOAT_EQ(-57.13902, stan::prob::exponential_log(15.0,3.9));
}
