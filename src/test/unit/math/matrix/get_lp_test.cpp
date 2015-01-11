#include <stan/math/matrix/accumulator.hpp>
#include <stan/math/matrix/get_lp.hpp>
#include <gtest/gtest.h>

TEST(mathMatrix,getLp) {
  using stan::math::accumulator;
  using stan::math::get_lp;
  
  double lp = 12.5;
  accumulator<double> lp_accum;
  EXPECT_FLOAT_EQ(12.5, get_lp(lp,lp_accum));

  lp_accum.add(2);
  lp_accum.add(3);
  EXPECT_FLOAT_EQ(17.5, get_lp(lp,lp_accum));
}

