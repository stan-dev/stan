#include <stan/math/matrix/min.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fvar.hpp>

using stan::agrad::fvar;
TEST(AgradFwdMatrix, min_vector) {
  using stan::math::min;
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
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrix, min_vector_exception) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d;
  vector_fv v;
  d.resize(0);
  v.resize(0);
  EXPECT_EQ(std::numeric_limits<double>::infinity(), min(v).val_);
  EXPECT_EQ(0, min(v).d_);
}
TEST(AgradFwdMatrix, min_rowvector) {
  using stan::math::min;
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
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrix, min_rowvector_exception) {
  using stan::math::min;
  using stan::agrad::row_vector_fv;

  row_vector_fv v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_);
  EXPECT_FLOAT_EQ(0, min(v).d_);
}
TEST(AgradFwdMatrix, min_matrix) {
  using stan::math::min;
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
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrix, min_matrix_exception) {
  using stan::math::min;
  using stan::agrad::matrix_fv;

  matrix_fv v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_);
  EXPECT_EQ(0, min(v).d_);
}
