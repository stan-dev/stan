#include <gtest/gtest.h>
#include "stan/prob/distributions_beta.hpp"

TEST(ProbDistributions,Beta) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::beta_log(0.2,1.0,1.0));
  EXPECT_FLOAT_EQ(1.628758, stan::prob::beta_log(0.3,12.0,25.0));
}

