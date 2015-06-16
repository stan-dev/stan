#include <stdexcept>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/sub_row.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixSubRow,SubRow1_matrix_fd) {
  using stan::math::sub_row;
  stan::math::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(0, sub_row(m,1,1,0).size());
}
TEST(AgradFwdMatrixSubRow,SubRow2_matrix_fd) {
  using stan::math::sub_row;
  stan::math::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(3, sub_row(m,1,1,3).size());
}
TEST(AgradFwdMatrixSubRow,SubRow3_matrix_fd) {
  using stan::math::sub_row;
  stan::math::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_row(m,1,1,7), std::out_of_range);
}
TEST(AgradFwdMatrixSubRow,SubRow4_matrix_fd) {
  using stan::math::sub_row;
  stan::math::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_row(m,7,1,1), std::out_of_range);
}
TEST(AgradFwdMatrixSubRow,SubRow5_matrix_fd) {
  using stan::math::sub_row;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_row(m,1,7,1), std::out_of_range);
}
TEST(AgradFwdMatrixSubRow,SubRow6_matrix_fd) {
  using stan::math::sub_row;
  stan::math::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) {
      m(i,j) = (i + 1) * (j + 1);
      m(i,j).d_ = 1.0;
    }
  stan::math::row_vector_fd v = sub_row(m,1,2,2);
  EXPECT_EQ(2,v.size());
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(m(0,1+i).val_, v(i).val_);
    EXPECT_FLOAT_EQ(m(0,1+i).d_, v(i).d_);
  }
}
TEST(AgradFwdMatrixSubRow,SubRow1_matrix_ffd) {
  using stan::math::sub_row;
  stan::math::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(0, sub_row(m,1,1,0).size());
}
TEST(AgradFwdMatrixSubRow,SubRow2_matrix_ffd) {
  using stan::math::sub_row;
  stan::math::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(3, sub_row(m,1,1,3).size());
}
TEST(AgradFwdMatrixSubRow,SubRow3_matrix_ffd) {
  using stan::math::sub_row;
  stan::math::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_row(m,1,1,7), std::out_of_range);
}
TEST(AgradFwdMatrixSubRow,SubRow4_matrix_ffd) {
  using stan::math::sub_row;
  stan::math::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_row(m,7,1,1), std::out_of_range);
}
TEST(AgradFwdMatrixSubRow,SubRow5_matrix_ffd) {
  using stan::math::sub_row;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_row(m,1,7,1), std::out_of_range);
}
TEST(AgradFwdMatrixSubRow,SubRow6_matrix_ffd) {
  using stan::math::sub_row;
  stan::math::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) {
      m(i,j) = (i + 1) * (j + 1);
      m(i,j).d_ = 1.0;
    }
  stan::math::row_vector_ffd v = sub_row(m,1,2,2);
  EXPECT_EQ(2,v.size());
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(m(0,1+i).val_.val_, v(i).val_.val_);
    EXPECT_FLOAT_EQ(m(0,1+i).d_.val_, v(i).d_.val_);
  }
}
