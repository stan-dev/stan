#include <stdexcept>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/sub_col.hpp>
#include <gtest/gtest.h>


TEST(AgradFwdMatrixSubCol,SubCol1_matrix_fd) {
  using stan::math::sub_col;
  stan::math::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(0, sub_col(m,1,1,0).size());
}
TEST(AgradFwdMatrixSubCol,SubCol2_matrix_fd) {
  using stan::math::sub_col;
  stan::math::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(3, sub_col(m,1,1,3).size());
}
TEST(AgradFwdMatrixSubCol,SubCol3_matrix_fd) {
  using stan::math::sub_col;
  stan::math::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,1,7), std::out_of_range);
}
TEST(AgradFwdMatrixSubCol,SubCol4_matrix_fd) {
  using stan::math::sub_col;
  stan::math::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,7,1,1), std::out_of_range);
}
TEST(AgradFwdMatrixSubCol,SubCol5_matrix_fd) {
  using stan::math::sub_col;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,7,1), std::out_of_range);
}
TEST(AgradFwdMatrixSubCol,SubCol6_matrix_fd) {
  using stan::math::sub_col;
  stan::math::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) {
      m(i,j) = (i + 1) * (j + 1);
      m(i,j).d_ = 1.0;
    }
  stan::math::row_vector_fd v = sub_col(m,1,2,2);
  EXPECT_EQ(2,v.size());
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(m(0+i,1).val_, v(i).val_);
    EXPECT_FLOAT_EQ(m(0+i,1).d_, v(i).d_);
  }
}
TEST(AgradFwdMatrixSubCol,SubCol1_matrix_ffd) {
  using stan::math::sub_col;
  stan::math::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(0, sub_col(m,1,1,0).size());
}
TEST(AgradFwdMatrixSubCol,SubCol2_matrix_ffd) {
  using stan::math::sub_col;
  stan::math::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(3, sub_col(m,1,1,3).size());
}
TEST(AgradFwdMatrixSubCol,SubCol3_matrix_ffd) {
  using stan::math::sub_col;
  stan::math::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,1,7), std::out_of_range);
}
TEST(AgradFwdMatrixSubCol,SubCol4_matrix_ffd) {
  using stan::math::sub_col;
  stan::math::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,7,1,1), std::out_of_range);
}
TEST(AgradFwdMatrixSubCol,SubCol5_matrix_ffd) {
  using stan::math::sub_col;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,7,1), std::out_of_range);
}
TEST(AgradFwdMatrixSubCol,SubCol6_matrix_ffd) {
  using stan::math::sub_col;
  stan::math::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) {
      m(i,j) = (i + 1) * (j + 1);
      m(i,j).d_ = 1.0;
    }
  stan::math::row_vector_ffd v = sub_col(m,1,2,2);
  EXPECT_EQ(2,v.size());
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(m(0+i,1).val_.val_, v(i).val_.val_);
    EXPECT_FLOAT_EQ(m(0+i,1).d_.val_, v(i).d_.val_);
  }
}
