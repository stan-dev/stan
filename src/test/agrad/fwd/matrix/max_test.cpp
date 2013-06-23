#include <stan/math/matrix/max.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fvar.hpp>

using stan::agrad::fvar;
TEST(AgradFwdMatrix, max_vector) {
  using stan::math::max;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<double> output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrix, max_vector_exception) {
  using stan::math::max;
  using stan::agrad::vector_fv;

  vector_fv v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_);
  EXPECT_EQ(0, max(v).d_);
}
TEST(AgradFwdMatrix, max_rowvector) {
  using stan::math::max;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<double> output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrix, max_rowvector_exception) {
  using stan::math::max;
  using stan::agrad::row_vector_fv;

  row_vector_fv v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_);
  EXPECT_EQ(0, max(v).d_);
}
TEST(AgradFwdMatrix, max_matrix) {
  using stan::math::max;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d d1(3,1);
  matrix_fv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<double> output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrix, max_matrix_exception) {
  using stan::math::max;
  using stan::agrad::matrix_fv;
  
  matrix_fv v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_);
  EXPECT_EQ(0, max(v).d_);
}
