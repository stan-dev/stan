#include <gtest/gtest.h>
#include "stan/prob/distributions_neg_binomial.hpp"

TEST(distributions,NegBinomial) {
  EXPECT_FLOAT_EQ(-7.786663, stan::prob::neg_binomial_log(10,2.0,1.5));
  EXPECT_FLOAT_EQ(-142.6147, stan::prob::neg_binomial_log(100,3.0,3.5));
}
