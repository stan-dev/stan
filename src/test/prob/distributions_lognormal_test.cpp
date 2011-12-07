#include <gtest/gtest.h>
#include "stan/prob/distributions.hpp"

TEST(prob_prob,lognormal) {
  EXPECT_FLOAT_EQ(-1.509803, stan::prob::lognormal_log(1.2,0.3,1.5));
  EXPECT_FLOAT_EQ(-3.462263, stan::prob::lognormal_log(12.0,3.0,0.9));
}
