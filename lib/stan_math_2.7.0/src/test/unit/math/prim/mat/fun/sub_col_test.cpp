#include <stdexcept>
#include <stan/math/prim/mat/fun/sub_col.hpp>
#include <gtest/gtest.h>


TEST(MathMatrixSubCol,SubCol1) {
  using stan::math::sub_col;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(0, sub_col(m,1,1,0).size());
}
TEST(MathMatrixSubCol,SubCol2) {
  using stan::math::sub_col;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(3, sub_col(m,1,1,3).size());
}
TEST(MathMatrixSubCol,SubCol3) {
  using stan::math::sub_col;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,1,7), std::out_of_range);
}
TEST(MathMatrixSubCol,SubCol4) {
  using stan::math::sub_col;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,7,1,1), std::out_of_range);
}
TEST(MathMatrixSubCol,SubCol5) {
  using stan::math::sub_col;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,7,1), std::out_of_range);
}
TEST(MathMatrixSubCol,SubCol6) {
  using stan::math::sub_col;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  Eigen::VectorXd v = sub_col(m,1,2,2);
  EXPECT_EQ(2,v.size());
  for (int i = 0; i < 2; ++i)
    EXPECT_FLOAT_EQ(m(0+i,1), v(i));
}
