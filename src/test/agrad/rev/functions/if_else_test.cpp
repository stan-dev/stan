#include <stan/agrad/rev/functions/if_else.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/functions/if_else.hpp>

TEST(AgradRev,if_else) {
  using stan::agrad::var;
  using stan::math::if_else;
  using stan::agrad::if_else;
  
  EXPECT_FLOAT_EQ(1.0,if_else(true,var(1.0),var(2.0)).val());
  EXPECT_FLOAT_EQ(2.0,if_else(false,var(1.0),var(2.0)).val());

  EXPECT_FLOAT_EQ(1.0,if_else(true,1.0,var(2.0)).val());
  EXPECT_FLOAT_EQ(2.0,if_else(false,1.0,var(2.0)).val());

  EXPECT_FLOAT_EQ(1.0,if_else(true,var(1.0),2.0).val());
  EXPECT_FLOAT_EQ(2.0,if_else(false,var(1.0),2.0).val());
}
