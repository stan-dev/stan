#include <stdexcept>
#include <stan/math/prim/mat/fun/sub_row.hpp>
#include <gtest/gtest.h>


TEST(MathMatrixSubRow,SubRow1) {
  using stan::math::sub_row;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(0, sub_row(m,1,1,0).size());
}
TEST(MathMatrixSubRow,SubRow2) {
  using stan::math::sub_row;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(3, sub_row(m,1,1,3).size());
}
TEST(MathMatrixSubRow,SubRow3) {
  using stan::math::sub_row;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_row(m,1,1,7), std::out_of_range);
}
TEST(MathMatrixSubRow,SubRow4) {
  using stan::math::sub_row;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_row(m,7,1,1), std::out_of_range);
}
TEST(MathMatrixSubRow,SubRow5) {
  using stan::math::sub_row;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_row(m,1,7,1), std::out_of_range);
}
TEST(MathMatrixSubRow,SubRow6) {
  using stan::math::sub_row;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  Eigen::RowVectorXd v = sub_row(m,1,2,2);
  EXPECT_EQ(2,v.size());
  for (int i = 0; i < 2; ++i)
    EXPECT_FLOAT_EQ(m(0,1+i), v(i));
}
