#include <stan/math/prim/mat/fun/transpose.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>

TEST(AgradFwdMatrixTranspose,fd_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::transpose;

  EXPECT_EQ(0,transpose(matrix_fd()).size());
  EXPECT_EQ(0,transpose(matrix_d()).size());

  matrix_fd a(2,3);
  a << -1.0, 2.0, -3.0, 
    5.0, 10.0, 100.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(0,2).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;
   a(1,2).d_ = 1.0;
  
  matrix_fd c = transpose(a);
  EXPECT_FLOAT_EQ(-1.0,c(0,0).val_);
  EXPECT_FLOAT_EQ( 5.0,c(0,1).val_);
  EXPECT_FLOAT_EQ( 2.0,c(1,0).val_);
  EXPECT_FLOAT_EQ(10.0,c(1,1).val_);
  EXPECT_FLOAT_EQ(-3.0,c(2,0).val_);
  EXPECT_FLOAT_EQ(100.0,c(2,1).val_);
  EXPECT_FLOAT_EQ( 1.0,c(0,0).d_);
  EXPECT_FLOAT_EQ( 1.0,c(0,1).d_);
  EXPECT_FLOAT_EQ( 1.0,c(1,0).d_);
  EXPECT_FLOAT_EQ( 1.0,c(1,1).d_);
  EXPECT_FLOAT_EQ( 1.0,c(2,0).d_);
  EXPECT_FLOAT_EQ( 1.0,c(2,1).d_);
  EXPECT_EQ(3,c.rows());
  EXPECT_EQ(2,c.cols());

}
TEST(AgradFwdMatrixTranspose,fd_vector) {
  using stan::math::vector_fd;
  using stan::math::row_vector_fd;
  using stan::math::transpose;
  using stan::math::size_type;

  vector_fd a(3);
  a << 1.0, 2.0, 3.0;
   a(0).d_ = 1.0;
   a(1).d_ = 1.0;
   a(2).d_ = 1.0;
  
  row_vector_fd a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_type i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val_,a_tr(i).val_);
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_);
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_);
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_);
}
TEST(AgradFwdMatrixTranspose,fd_row_vector) {
  using stan::math::vector_fd;
  using stan::math::row_vector_fd;
  using stan::math::transpose;
  using stan::math::size_type;

  row_vector_fd a(3);
  a << 1.0, 2.0, 3.0;
   a(0).d_ = 1.0;
   a(1).d_ = 1.0;
   a(2).d_ = 1.0;
  
  vector_fd a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_type i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val_,a_tr(i).val_);
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_);
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_);
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_);
}
TEST(AgradFwdMatrixTranspose,ffd_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::transpose;

  EXPECT_EQ(0,transpose(matrix_ffd()).size());
  EXPECT_EQ(0,transpose(matrix_d()).size());

  matrix_ffd a(2,3);
  a << -1.0, 2.0, -3.0, 
    5.0, 10.0, 100.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(0,2).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;
   a(1,2).d_ = 1.0;
  
  matrix_ffd c = transpose(a);
  EXPECT_FLOAT_EQ(-1.0,c(0,0).val_.val());
  EXPECT_FLOAT_EQ( 5.0,c(0,1).val_.val());
  EXPECT_FLOAT_EQ( 2.0,c(1,0).val_.val());
  EXPECT_FLOAT_EQ(10.0,c(1,1).val_.val());
  EXPECT_FLOAT_EQ(-3.0,c(2,0).val_.val());
  EXPECT_FLOAT_EQ(100.0,c(2,1).val_.val());
  EXPECT_FLOAT_EQ( 1.0,c(0,0).d_.val());
  EXPECT_FLOAT_EQ( 1.0,c(0,1).d_.val());
  EXPECT_FLOAT_EQ( 1.0,c(1,0).d_.val());
  EXPECT_FLOAT_EQ( 1.0,c(1,1).d_.val());
  EXPECT_FLOAT_EQ( 1.0,c(2,0).d_.val());
  EXPECT_FLOAT_EQ( 1.0,c(2,1).d_.val());
  EXPECT_EQ(3,c.rows());
  EXPECT_EQ(2,c.cols());

}
TEST(AgradFwdMatrixTranspose,ffd_vector) {
  using stan::math::vector_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::transpose;
  using stan::math::size_type;

  vector_ffd a(3);
  a << 1.0, 2.0, 3.0;
   a(0).d_ = 1.0;
   a(1).d_ = 1.0;
   a(2).d_ = 1.0;
  
  row_vector_ffd a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_type i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val_.val(),a_tr(i).val_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_.val());
}
TEST(AgradFwdMatrixTranspose,ffd_row_vector) {
  using stan::math::vector_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::transpose;
  using stan::math::size_type;

  row_vector_ffd a(3);
  a << 1.0, 2.0, 3.0;
   a(0).d_ = 1.0;
   a(1).d_ = 1.0;
   a(2).d_ = 1.0;
  
  vector_ffd a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_type i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val_.val(),a_tr(i).val_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_.val());
}
