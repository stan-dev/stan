#include <stan/math/matrix/to_vector.hpp>
#include <stan/agrad/var.hpp>
#include <gtest/gtest.h>

TEST(AgradRevMatrix, to_vector) {
  using stan::math::to_vector;
  using stan::agrad::var;

  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> a(3,3);
  Eigen::Matrix<var,Eigen::Dynamic,1> b(9);

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      a(j,i) = j + i;
    }
  }

  b = to_vector(a);

  EXPECT_FLOAT_EQ(0,b(0).val());
  EXPECT_FLOAT_EQ(1,b(1).val());
  EXPECT_FLOAT_EQ(2,b(2).val());
  EXPECT_FLOAT_EQ(1,b(3).val());
  EXPECT_FLOAT_EQ(2,b(4).val());
  EXPECT_FLOAT_EQ(3,b(5).val());
  EXPECT_FLOAT_EQ(2,b(6).val());
  EXPECT_FLOAT_EQ(3,b(7).val());
  EXPECT_FLOAT_EQ(4,b(8).val());
}
