#include <gtest/gtest.h>
#include <stan/math/rep_vector.hpp>

TEST(MathMatrix,rep_vector) {
  using stan::math::rep_vector;
  Eigen::Matrix<double,Eigen::Dynamic,1> x = rep_vector(2.0, 3);
  EXPECT_EQ(3,x.size());
  for (int i = 0; i < x.size(); ++i)
    EXPECT_FLOAT_EQ(2.0, x[i]);

  EXPECT_THROW(rep_vector(2.0,-1), std::domain_error);
}
