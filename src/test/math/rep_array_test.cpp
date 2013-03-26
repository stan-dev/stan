#include <gtest/gtest.h>
#include <stan/math/rep_array.hpp>

TEST(MathMatrix,rep_array) {
  using stan::math::rep_array;
  std::vector<double> x = rep_array(2.0, 3);
  EXPECT_EQ(3,x.size());
  for (size_t i = 0; i < x.size(); ++i)
    EXPECT_FLOAT_EQ(2.0, x[i]);

  EXPECT_THROW(rep_array(2.0,-1), std::domain_error);
}
