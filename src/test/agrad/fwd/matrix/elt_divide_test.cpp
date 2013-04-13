#include <stan/math/matrix/elt_divide.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

TEST(AgradFwdMatrix,elt_divide_vec_vv) {
  using stan::math::elt_divide;
  using stan::agrad::vector_fv;

  vector_fv x(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  vector_fv y(2);
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_);
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_);

}
TEST(AgradFwdMatrix,elt_divide_vec_vd) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_fv x(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  vector_d y(2);
  y << 10, 100;

  vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(0.1,z(0).d_);
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_);
}
TEST(AgradFwdMatrix,elt_divide_vec_dv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d x(2);
  x << 2, 5;
  vector_fv y(2);
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(-0.02,z(0).d_);
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_);
}

TEST(AgradFwdMatrix,elt_divide_rowvec_vv) {
  using stan::math::elt_divide;
  using stan::agrad::row_vector_fv;

  row_vector_fv x(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  row_vector_fv y(2);
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  row_vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_);
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_);
}
TEST(AgradFwdMatrix,elt_divide_rowvec_vd) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_fv x(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  row_vector_d y(2);
  y << 10, 100;

  row_vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(0.1,z(0).d_);
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_);
}
TEST(AgradFwdMatrix,elt_divide_rowvec_dv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d x(2);
  x << 2, 5;
  row_vector_fv y(2);
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  row_vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(-0.02,z(0).d_);
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_);
}

TEST(AgradFwdMatrix,elt_divide_mat_vv) {
  using stan::math::elt_divide;
  using stan::agrad::matrix_fv;

  matrix_fv x(2,3);
  x << 2, 5, 7, 13, 29, 112;
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

  matrix_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_);
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_);
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_);
  EXPECT_FLOAT_EQ(0.079999998,z(0,0).d_);
  EXPECT_FLOAT_EQ(0.0094999997,z(0,1).d_);
  EXPECT_FLOAT_EQ(9.9988802e-07,z(1,2).d_);
}
TEST(AgradFwdMatrix,elt_divide_mat_vd) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  
  matrix_fv x(2,3);
  x << 2, 5, 7, 13, 29, 112;
   x(0,0).d_ = 1.0;
   x(0,1).d_ = 1.0;
   x(0,2).d_ = 1.0;
   x(1,0).d_ = 1.0;
   x(1,1).d_ = 1.0;
   x(1,2).d_ = 1.0;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;

  matrix_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_);
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_);
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_);
  EXPECT_FLOAT_EQ(0.1,z(0,0).d_);
  EXPECT_FLOAT_EQ(0.0099999998,z(0,1).d_);
  EXPECT_FLOAT_EQ(1e-06,z(1,2).d_);
}
TEST(AgradFwdMatrix,elt_divide_mat_dv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d x(2,3);
  x << 2, 5, 7, 13, 29, 112;
  matrix_fv y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;

  matrix_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_);
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_);
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_);
  EXPECT_FLOAT_EQ(-0.02,z(0,0).d_);
  EXPECT_FLOAT_EQ(-0.00050000002,z(0,1).d_);
  EXPECT_FLOAT_EQ(-1.12e-10,z(1,2).d_);
}
