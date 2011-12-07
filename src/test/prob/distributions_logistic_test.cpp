#include <gtest/gtest.h>
#include "stan/prob/distributions_logistic.hpp"

TEST(distributions,Logistic) {
  EXPECT_FLOAT_EQ(-2.129645, stan::prob::logistic_log(1.2,0.3,2.0));
  EXPECT_FLOAT_EQ(-3.430098, stan::prob::logistic_log(-1.0,0.2,0.25));
}
