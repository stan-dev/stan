#include <stan/agrad/rev/var.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,a_eq_x) {
  stan::agrad::var a = 5.0;
  EXPECT_FLOAT_EQ(5.0,a.val());
}

TEST(AgradRev,a_of_x) {
  stan::agrad::var a(6.0);
  EXPECT_FLOAT_EQ(6.0,a.val());
}

TEST(AgradRev,a__a_eq_x) {
  stan::agrad::var a;
  a = 7.0;
  EXPECT_FLOAT_EQ(7.0,a.val());
}
