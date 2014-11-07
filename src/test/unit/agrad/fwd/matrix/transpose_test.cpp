#include <stan/math/matrix/transpose.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixTranspose,fd_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;
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
  using stan::agrad::vector_fd;
  using stan::agrad::row_vector_fd;
  using stan::math::transpose;
  using stan::agrad::size_type;

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
  using stan::agrad::vector_fd;
  using stan::agrad::row_vector_fd;
  using stan::math::transpose;
  using stan::agrad::size_type;

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
TEST(AgradFwdMatrixTranspose,fv_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::math::transpose;

  EXPECT_EQ(0,transpose(matrix_fv()).size());
  EXPECT_EQ(0,transpose(matrix_d()).size());

  matrix_fv a(2,3);
  a << -1.0, 2.0, -3.0, 
    5.0, 10.0, 100.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(0,2).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;
   a(1,2).d_ = 1.0;
  
  matrix_fv c = transpose(a);
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
TEST(AgradFwdMatrixTranspose,fv_vector) {
  using stan::agrad::vector_fv;
  using stan::agrad::row_vector_fv;
  using stan::math::transpose;
  using stan::agrad::size_type;

  vector_fv a(3);
  a << 1.0, 2.0, 3.0;
   a(0).d_ = 1.0;
   a(1).d_ = 1.0;
   a(2).d_ = 1.0;
  
  row_vector_fv a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_type i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val_.val(),a_tr(i).val_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_.val());
}
TEST(AgradFwdMatrixTranspose,fv_row_vector) {
  using stan::agrad::vector_fv;
  using stan::agrad::row_vector_fv;
  using stan::math::transpose;
  using stan::agrad::size_type;

  row_vector_fv a(3);
  a << 1.0, 2.0, 3.0;
   a(0).d_ = 1.0;
   a(1).d_ = 1.0;
   a(2).d_ = 1.0;
  
  vector_fv a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_type i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val_.val(),a_tr(i).val_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_.val());
}
TEST(AgradFwdMatrixTranspose,ffd_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
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
  using stan::agrad::vector_ffd;
  using stan::agrad::row_vector_ffd;
  using stan::math::transpose;
  using stan::agrad::size_type;

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
  using stan::agrad::vector_ffd;
  using stan::agrad::row_vector_ffd;
  using stan::math::transpose;
  using stan::agrad::size_type;

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

TEST(AgradFwdMatrixTranspose,ffv_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::math::transpose;

  EXPECT_EQ(0,transpose(matrix_ffv()).size());
  EXPECT_EQ(0,transpose(matrix_d()).size());

  matrix_ffv a(2,3);
  a << -1.0, 2.0, -3.0, 
    5.0, 10.0, 100.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(0,2).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;
   a(1,2).d_ = 1.0;
  
  matrix_ffv c = transpose(a);
  EXPECT_FLOAT_EQ(-1.0,c(0,0).val_.val().val());
  EXPECT_FLOAT_EQ( 5.0,c(0,1).val_.val().val());
  EXPECT_FLOAT_EQ( 2.0,c(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(10.0,c(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(-3.0,c(2,0).val_.val().val());
  EXPECT_FLOAT_EQ(100.0,c(2,1).val_.val().val());
  EXPECT_FLOAT_EQ( 1.0,c(0,0).d_.val().val());
  EXPECT_FLOAT_EQ( 1.0,c(0,1).d_.val().val());
  EXPECT_FLOAT_EQ( 1.0,c(1,0).d_.val().val());
  EXPECT_FLOAT_EQ( 1.0,c(1,1).d_.val().val());
  EXPECT_FLOAT_EQ( 1.0,c(2,0).d_.val().val());
  EXPECT_FLOAT_EQ( 1.0,c(2,1).d_.val().val());
  EXPECT_EQ(3,c.rows());
  EXPECT_EQ(2,c.cols());

}
TEST(AgradFwdMatrixTranspose,ffv_vector) {
  using stan::agrad::vector_ffv;
  using stan::agrad::row_vector_ffv;
  using stan::math::transpose;
  using stan::agrad::size_type;

  vector_ffv a(3);
  a << 1.0, 2.0, 3.0;
   a(0).d_ = 1.0;
   a(1).d_ = 1.0;
   a(2).d_ = 1.0;
  
  row_vector_ffv a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_type i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val_.val().val(),a_tr(i).val_.val().val());
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_.val().val());
}
TEST(AgradFwdMatrixTranspose,ffv_row_vector) {
  using stan::agrad::vector_ffv;
  using stan::agrad::row_vector_ffv;
  using stan::math::transpose;
  using stan::agrad::size_type;

  row_vector_ffv a(3);
  a << 1.0, 2.0, 3.0;
   a(0).d_ = 1.0;
   a(1).d_ = 1.0;
   a(2).d_ = 1.0;
  
  vector_ffv a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_type i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val_.val().val(),a_tr(i).val_.val().val());
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_.val().val());
}
