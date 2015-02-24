#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/int_step.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradRev,int_step) {
  using stan::math::int_step;

  AVAR a(5.0);
  AVAR b(0.0);
  AVAR c(-1.0);
  
  EXPECT_EQ(1U,int_step(a));
  EXPECT_EQ(0U,int_step(b));
  EXPECT_EQ(0U,int_step(c));
}
