#include <stan/agrad/rev.hpp>
#include <stan/math.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>


TEST(AgradRev,int_step) {
  using stan::math::int_step;

  AVAR a(5.0);
  AVAR b(0.0);
  AVAR c(-1.0);
  
  EXPECT_EQ(1U,int_step(a));
  EXPECT_EQ(0U,int_step(b));
  EXPECT_EQ(0U,int_step(c));
}
