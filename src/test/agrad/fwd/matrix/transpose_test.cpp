#include <stan/math/matrix/transpose.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/var.hpp>

TEST(AgradFwdMatrix,transpose_matrix) {
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
TEST(AgradFwdMatrix,transpose_vector) {
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
    EXPECT_FLOAT_EQ(a(i).val_,a_tr(i).val_);
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_);
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_);
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_);
}
TEST(AgradFwdMatrix,transpose_row_vector) {
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
    EXPECT_FLOAT_EQ(a(i).val_,a_tr(i).val_);
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_);
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_);
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_);
}
TEST(AgradFwdFvarVarMatrix,transpose_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fvv;
  using stan::math::transpose;

  EXPECT_EQ(0,transpose(matrix_fvv()).size());
  EXPECT_EQ(0,transpose(matrix_d()).size());

  matrix_fvv a(2,3);
  a << -1.0, 2.0, -3.0, 
    5.0, 10.0, 100.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(0,2).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;
   a(1,2).d_ = 1.0;
  
  matrix_fvv c = transpose(a);
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
TEST(AgradFwdFvarVarMatrix,transpose_vector) {
  using stan::agrad::vector_fvv;
  using stan::agrad::row_vector_fvv;
  using stan::math::transpose;
  using stan::agrad::size_type;

  vector_fvv a(3);
  a << 1.0, 2.0, 3.0;
   a(0).d_ = 1.0;
   a(1).d_ = 1.0;
   a(2).d_ = 1.0;
  
  row_vector_fvv a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_type i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val_.val(),a_tr(i).val_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_.val());
}
TEST(AgradFwdFvarVarMatrix,transpose_row_vector) {
  using stan::agrad::vector_fvv;
  using stan::agrad::row_vector_fvv;
  using stan::math::transpose;
  using stan::agrad::size_type;

  row_vector_fvv a(3);
  a << 1.0, 2.0, 3.0;
   a(0).d_ = 1.0;
   a(1).d_ = 1.0;
   a(2).d_ = 1.0;
  
  vector_fvv a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_type i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val_.val(),a_tr(i).val_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_.val());
}
TEST(AgradFwdFvarFvarMatrix,transpose_matrix) {
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
TEST(AgradFwdFvarFvarMatrix,transpose_vector) {
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
    EXPECT_FLOAT_EQ(a(i).val_.val(),a_tr(i).val_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_.val());
}
TEST(AgradFwdFvarFvarMatrix,transpose_row_vector) {
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
    EXPECT_FLOAT_EQ(a(i).val_.val(),a_tr(i).val_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(0).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(1).d_.val());
  EXPECT_FLOAT_EQ(1.0, a_tr(2).d_.val());
}
