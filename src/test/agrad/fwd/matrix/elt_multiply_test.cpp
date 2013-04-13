#include <stan/math/matrix/elt_multiply.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

TEST(AgradFwdMatrix,elt_multiply_vec_vv) {
  using stan::math::elt_multiply;
  using stan::agrad::vector_fv;

  vector_fv x(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  vector_fv y(2);
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  vector_fv z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val_);
  EXPECT_FLOAT_EQ(500.0,z(1).val_);
  EXPECT_FLOAT_EQ(12,z(0).d_);
  EXPECT_FLOAT_EQ(105,z(1).d_);
}

TEST(AgradFwdMatrix,elt_multiply_vec_vd) {
  using stan::math::elt_multiply;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_fv x(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  vector_d y(2);
  y << 10, 100;

  vector_fv z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val_);
  EXPECT_FLOAT_EQ(500.0,z(1).val_);
  EXPECT_FLOAT_EQ(10,z(0).d_);
  EXPECT_FLOAT_EQ(100,z(1).d_);
}
TEST(AgradFwdMatrix,elt_multiply_vec_dv) {
  using stan::math::elt_multiply;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d x(2);
  x << 2, 5;
  vector_fv y(2);
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  vector_fv z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val_);
  EXPECT_FLOAT_EQ(500.0,z(1).val_);
  EXPECT_FLOAT_EQ(2,z(0).d_);
  EXPECT_FLOAT_EQ(5,z(1).d_);
}

TEST(AgradFwdMatrix,elt_multiply_row_vec_vv) {
  using stan::math::elt_multiply;
  using stan::agrad::row_vector_fv;

  row_vector_fv x(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  row_vector_fv y(2);
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  row_vector_fv z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val_);
  EXPECT_FLOAT_EQ(500.0,z(1).val_);
  EXPECT_FLOAT_EQ(12,z(0).d_);
  EXPECT_FLOAT_EQ(105,z(1).d_);
}
TEST(AgradFwdMatrix,elt_multiply_row_vec_vd) {
  using stan::math::elt_multiply;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_fv x(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  row_vector_d y(2);
  y << 10, 100;

  row_vector_fv z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val_);
  EXPECT_FLOAT_EQ(500.0,z(1).val_);
  EXPECT_FLOAT_EQ(10,z(0).d_);
  EXPECT_FLOAT_EQ(100,z(1).d_);
}
TEST(AgradFwdMatrix,elt_multiply_row_vec_dv) {
  using stan::math::elt_multiply;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d x(2);
  x << 2, 5;
  row_vector_fv y(2);
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  row_vector_fv z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val_);
  EXPECT_FLOAT_EQ(500.0,z(1).val_);
  EXPECT_FLOAT_EQ(2,z(0).d_);
  EXPECT_FLOAT_EQ(5,z(1).d_);
}

TEST(AgradFwdMatrix,elt_multiply_matrix_vv) {
  using stan::math::elt_multiply;
  using stan::agrad::matrix_fv;

  matrix_fv x(2,3);
  x << 2, 5, 6, 9, 13, 29;
   x(0,0).d_ = 1.0;
   x(0,1).d_ = 1.0;
   x(0,2).d_ = 1.0;
   x(1,0).d_ = 1.0;
   x(1,1).d_ = 1.0;
   x(1,2).d_ = 1.0;
  matrix_fv y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;

  matrix_fv z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0,0).val_);
  EXPECT_FLOAT_EQ(500.0,z(0,1).val_);
  EXPECT_FLOAT_EQ(29000000.0,z(1,2).val_);
  EXPECT_FLOAT_EQ(12,z(0,0).d_);
  EXPECT_FLOAT_EQ(105,z(0,1).d_);
  EXPECT_FLOAT_EQ(1000029,z(1,2).d_);
}
TEST(AgradFwdMatrix,elt_multiply_matrix_vd) {
  using stan::math::elt_multiply;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_fv x(2,3);
  x << 2, 5, 6, 9, 13, 29;
   x(0,0).d_ = 1.0;
   x(0,1).d_ = 1.0;
   x(0,2).d_ = 1.0;
   x(1,0).d_ = 1.0;
   x(1,1).d_ = 1.0;
   x(1,2).d_ = 1.0;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;

  matrix_fv z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0,0).val_);
  EXPECT_FLOAT_EQ(500.0,z(0,1).val_);
  EXPECT_FLOAT_EQ(29000000.0,z(1,2).val_);
  EXPECT_FLOAT_EQ(10,z(0,0).d_);
  EXPECT_FLOAT_EQ(100,z(0,1).d_);
  EXPECT_FLOAT_EQ(1000000,z(1,2).d_);
}
TEST(AgradFwdMatrix,elt_multiply_matrix_dv) {
  using stan::math::elt_multiply;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d x(2,3);
  x << 2, 5, 6, 9, 13, 29;
  matrix_fv y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;

  matrix_fv z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0,0).val_);
  EXPECT_FLOAT_EQ(500.0,z(0,1).val_);
  EXPECT_FLOAT_EQ(29000000.0,z(1,2).val_);
  EXPECT_FLOAT_EQ(2,z(0,0).d_);
  EXPECT_FLOAT_EQ(5,z(0,1).d_);
  EXPECT_FLOAT_EQ(29,z(1,2).d_);
}
