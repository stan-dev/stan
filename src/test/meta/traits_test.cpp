#include <gtest/gtest.h>
#include <stan/meta/traits.hpp>
#include <stan/agrad/agrad.hpp>

TEST(traitsTest, isConstant) {
  EXPECT_TRUE(stan::is_constant<double>::value);
  EXPECT_FALSE(stan::is_constant<stan::agrad::var>::value);
}
