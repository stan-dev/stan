#include <stan/math/matrix/accumulator.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,accumulateDouble) {
  using stan::math::accumulator;
  
  accumulator<double> a;
  EXPECT_FLOAT_EQ(0.0, a.sum());

  a.add(1.0);
  EXPECT_FLOAT_EQ(1.0, a.sum());

  for (int i = 2; i <= 1000; ++i)
    a.add(i);
  
  EXPECT_FLOAT_EQ((1000 * 1001) / 2, a.sum());
  
}
