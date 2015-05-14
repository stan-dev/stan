#include <stan/math/prim/mat/fun/transpose.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>

TEST(AgradMixMatrixTranspose,fv_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
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
TEST(AgradMixMatrixTranspose,fv_vector) {
  using stan::math::vector_fv;
  using stan::math::row_vector_fv;
  using stan::math::transpose;
  using stan::math::size_type;

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
TEST(AgradMixMatrixTranspose,fv_row_vector) {
  using stan::math::vector_fv;
  using stan::math::row_vector_fv;
  using stan::math::transpose;
  using stan::math::size_type;

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
TEST(AgradMixMatrixTranspose,ffv_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
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
TEST(AgradMixMatrixTranspose,ffv_vector) {
  using stan::math::vector_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::transpose;
  using stan::math::size_type;

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
TEST(AgradMixMatrixTranspose,ffv_row_vector) {
  using stan::math::vector_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::transpose;
  using stan::math::size_type;

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
