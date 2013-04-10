#include <gtest/gtest.h>
#include <stan/math/rep_row_vector.hpp>

TEST(MathMatrix,rep_row_vector) {
  using stan::math::rep_row_vector;
  Eigen::Matrix<double,1,Eigen::Dynamic> x = rep_row_vector(2.0, 3);
  EXPECT_EQ(3,x.size());
  for (size_t i = 0; i < x.size(); ++i)
    EXPECT_FLOAT_EQ(2.0, x[i]);

}
