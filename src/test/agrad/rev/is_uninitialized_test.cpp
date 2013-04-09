#include <stan/agrad/rev/is_uninitialized.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,undefined) {
  stan::agrad::var a;
  EXPECT_TRUE(a.is_uninitialized());
  a = 5;
  EXPECT_FALSE(a.is_uninitialized());
}
