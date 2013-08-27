#include <stan/diff/fwd/matrix/sum.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/sum.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/diff/fwd/matrix/typedefs.hpp>
#include <stan/diff/fwd/fvar.hpp>

using stan::diff::fvar;

TEST(DiffFwdMatrix, sum_vector) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::diff::vector_fv;

  vector_d d(6);
  vector_fv v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
   v(5).d_ = 1.0;
  
  fvar<double> output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_);
  EXPECT_FLOAT_EQ( 0.0, output.d_);
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_);  
  EXPECT_FLOAT_EQ( 6.0, output.d_);

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_);
  EXPECT_FLOAT_EQ(0.0, sum(v).d_);
}
TEST(DiffFwdMatrix, sum_rowvector) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::diff::row_vector_fv;

  row_vector_d d(6);
  row_vector_fv v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
   v(5).d_ = 1.0;
  
  fvar<double> output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_);
  EXPECT_FLOAT_EQ( 0.0, output.d_);
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_);  
  EXPECT_FLOAT_EQ( 6.0, output.d_);

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_);
  EXPECT_FLOAT_EQ(0.0, sum(v).d_);
}
TEST(DiffFwdMatrix, sum_matrix) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::diff::matrix_fv;

  matrix_d d(2, 3);
  matrix_fv v(2, 3);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(0,2).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
   v(1,2).d_ = 1.0;
  
  fvar<double> output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_);
  EXPECT_FLOAT_EQ( 0.0, output.d_);
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_);
  EXPECT_FLOAT_EQ( 6.0, output.d_);

  d.resize(0, 0);
  v.resize(0, 0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_);
  EXPECT_FLOAT_EQ(0.0, sum(v).d_);
}
