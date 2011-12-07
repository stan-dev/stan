#include <gtest/gtest.h>
#include "stan/prob/distributions_poisson.hpp"

TEST(prob_prob,poisson) {
  EXPECT_FLOAT_EQ(-2.900934, stan::prob::poisson_log(17,13.0));
  EXPECT_FLOAT_EQ(-145.3547, stan::prob::poisson_log(192,42.0));
}
