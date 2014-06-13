#include <stdexcept>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/math/matrix/sub_col.hpp>
#include <gtest/gtest.h>


TEST(AgradFwdMatrixSubCol,SubCol1_matrix_fd) {
  using stan::math::sub_col;
  stan::agrad::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(0, sub_col(m,1,1,0).size());
}
TEST(AgradFwdMatrixSubCol,SubCol2_matrix_fd) {
  using stan::math::sub_col;
  stan::agrad::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(3, sub_col(m,1,1,3).size());
}
TEST(AgradFwdMatrixSubCol,SubCol3_matrix_fd) {
  using stan::math::sub_col;
  stan::agrad::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,1,7), std::domain_error);
}
TEST(AgradFwdMatrixSubCol,SubCol4_matrix_fd) {
  using stan::math::sub_col;
  stan::agrad::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,7,1,1), std::domain_error);
}
TEST(AgradFwdMatrixSubCol,SubCol5_matrix_fd) {
  using stan::math::sub_col;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,7,1), std::domain_error);
}
TEST(AgradFwdMatrixSubCol,SubCol6_matrix_fd) {
  using stan::math::sub_col;
  stan::agrad::matrix_fd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) {
      m(i,j) = (i + 1) * (j + 1);
      m(i,j).d_ = 1.0;
    }
  stan::agrad::row_vector_fd v = sub_col(m,1,2,2);
  EXPECT_EQ(2,v.size());
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(m(0+i,1).val_, v(i).val_);
    EXPECT_FLOAT_EQ(m(0+i,1).d_, v(i).d_);
  }
}
TEST(AgradFwdMatrixSubCol,SubCol1_matrix_fv) {
  using stan::math::sub_col;
  stan::agrad::matrix_fv m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(0, sub_col(m,1,1,0).size());
}
TEST(AgradFwdMatrixSubCol,SubCol2_matrix_fv) {
  using stan::math::sub_col;
  stan::agrad::matrix_fv m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(3, sub_col(m,1,1,3).size());
}
TEST(AgradFwdMatrixSubCol,SubCol3_matrix_fv) {
  using stan::math::sub_col;
  stan::agrad::matrix_fv m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,1,7), std::domain_error);
}
TEST(AgradFwdMatrixSubCol,SubCol4_matrix_fv) {
  using stan::math::sub_col;
  stan::agrad::matrix_fv m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,7,1,1), std::domain_error);
}
TEST(AgradFwdMatrixSubCol,SubCol5_matrix_fv) {
  using stan::math::sub_col;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,7,1), std::domain_error);
}
TEST(AgradFwdMatrixSubCol,SubCol6_matrix_fv) {
  using stan::math::sub_col;
  stan::agrad::matrix_fv m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) {
      m(i,j) = (i + 1) * (j + 1);
      m(i,j).d_ = 1.0;
    }
  stan::agrad::row_vector_fv v = sub_col(m,1,2,2);
  EXPECT_EQ(2,v.size());
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(m(0+i,1).val_.val(), v(i).val_.val());
    EXPECT_FLOAT_EQ(m(0+i,1).d_.val(), v(i).d_.val());
  }
}TEST(AgradFwdMatrixSubCol,SubCol1_matrix_ffd) {
  using stan::math::sub_col;
  stan::agrad::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(0, sub_col(m,1,1,0).size());
}
TEST(AgradFwdMatrixSubCol,SubCol2_matrix_ffd) {
  using stan::math::sub_col;
  stan::agrad::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(3, sub_col(m,1,1,3).size());
}
TEST(AgradFwdMatrixSubCol,SubCol3_matrix_ffd) {
  using stan::math::sub_col;
  stan::agrad::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,1,7), std::domain_error);
}
TEST(AgradFwdMatrixSubCol,SubCol4_matrix_ffd) {
  using stan::math::sub_col;
  stan::agrad::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,7,1,1), std::domain_error);
}
TEST(AgradFwdMatrixSubCol,SubCol5_matrix_ffd) {
  using stan::math::sub_col;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,7,1), std::domain_error);
}
TEST(AgradFwdMatrixSubCol,SubCol6_matrix_ffd) {
  using stan::math::sub_col;
  stan::agrad::matrix_ffd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) {
      m(i,j) = (i + 1) * (j + 1);
      m(i,j).d_ = 1.0;
    }
  stan::agrad::row_vector_ffd v = sub_col(m,1,2,2);
  EXPECT_EQ(2,v.size());
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(m(0+i,1).val_.val_, v(i).val_.val_);
    EXPECT_FLOAT_EQ(m(0+i,1).d_.val_, v(i).d_.val_);
  }
}
TEST(AgradFwdMatrixSubCol,SubCol1_matrix_ffv) {
  using stan::math::sub_col;
  stan::agrad::matrix_ffv m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(0, sub_col(m,1,1,0).size());
}
TEST(AgradFwdMatrixSubCol,SubCol2_matrix_ffv) {
  using stan::math::sub_col;
  stan::agrad::matrix_ffv m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_EQ(3, sub_col(m,1,1,3).size());
}
TEST(AgradFwdMatrixSubCol,SubCol3_matrix_ffv) {
  using stan::math::sub_col;
  stan::agrad::matrix_ffv m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,1,7), std::domain_error);
}
TEST(AgradFwdMatrixSubCol,SubCol4_matrix_ffv) {
  using stan::math::sub_col;
  stan::agrad::matrix_ffv m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,7,1,1), std::domain_error);
}
TEST(AgradFwdMatrixSubCol,SubCol5_matrix_ffv) {
  using stan::math::sub_col;
  Eigen::MatrixXd m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);
  EXPECT_THROW(sub_col(m,1,7,1), std::domain_error);
}
TEST(AgradFwdMatrixSubCol,SubCol6_matrix_ffv) {
  using stan::math::sub_col;
  stan::agrad::matrix_ffv m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) {
      m(i,j) = (i + 1) * (j + 1);
      m(i,j).d_ = 1.0;
    }
  stan::agrad::row_vector_ffv v = sub_col(m,1,2,2);
  EXPECT_EQ(2,v.size());
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(m(0+i,1).val_.val_.val(), v(i).val_.val_.val());
    EXPECT_FLOAT_EQ(m(0+i,1).d_.val_.val(), v(i).d_.val_.val());
  }
}




