#include <stan/agrad/fwd/matrix/mdivide_left_tri_low.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

TEST(AgradFwdMatrix, mdivide_left_tri_low_vector_fv_matrix_fv) {
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;

  matrix_fv Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
   Y(2,2).d_ = 2.0;

   vector_fv Z(3);
   Z << 1, 2, 3;
    Z(0).d_ = 2.0;
    Z(1).d_ = 2.0;
    Z(2).d_ = 2.0;

  matrix_fv output = stan::agrad::mdivide_left_tri_low(Y,Z);

  EXPECT_FLOAT_EQ(1.0,output(0,0).val_);
  EXPECT_FLOAT_EQ(1.110223e-16,output(1,0).val_);
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(2,0).val_);
  EXPECT_FLOAT_EQ( 1.8503717e-16,output(0,0).d_);
  EXPECT_FLOAT_EQ(-1.6653345e-16,output(1,0).d_);
  EXPECT_FLOAT_EQ(5.0 / 90.0,output(2,0).d_);
}
TEST(AgradFwdMatrix, mdivide_left_tri_low_vector_matrix_fv) {
  using stan::agrad::matrix_fv;
  using stan::math::vector_d;

  matrix_fv Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
   Y(2,2).d_ = 2.0;

   vector_d Z(3);
   Z << 1, 2, 3;

  matrix_fv output = stan::agrad::mdivide_left_tri_low(Y,Z);

  EXPECT_FLOAT_EQ(1.0,output(0,0).val_);
  EXPECT_FLOAT_EQ(1.110223e-16,output(1,0).val_);
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(2,0).val_);
  EXPECT_FLOAT_EQ(-2.0,output(0,0).d_);
  EXPECT_FLOAT_EQ(4.0 / 6.0,output(1,0).d_);
  EXPECT_FLOAT_EQ(1.0 / 2.0,output(2,0).d_);
}
TEST(AgradFwdMatrix, mdivide_left_tri_low_vector_fv_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  vector_fv Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 2.0;
   Z(1).d_ = 2.0;
   Z(2).d_ = 2.0;

  matrix_fv output = stan::agrad::mdivide_left_tri_low(Y,Z);

  EXPECT_FLOAT_EQ(1.0,output(0,0).val_);
  EXPECT_FLOAT_EQ(1.110223e-16,output(1,0).val_);
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(2,0).val_);
  EXPECT_FLOAT_EQ(2.0,output(0,0).d_);
  EXPECT_FLOAT_EQ(-4.0 / 6.0,output(1,0).d_);
  EXPECT_FLOAT_EQ(-4.0 / 9.0,output(2,0).d_);
}
TEST(AgradFwdMatrix, mdivide_left_tri_low_matrix_fv_matrix_fv) {
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;

  matrix_fv Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
   Y(2,2).d_ = 2.0;

  matrix_fv Z(3,3);
  Z << 1, 2, 3,
    6, 5, 4,
    7, 8, 9;
   Z(0,0).d_ = 2.0;
   Z(0,1).d_ = 2.0;
   Z(0,2).d_ = 2.0;
   Z(1,0).d_ = 2.0;
   Z(1,1).d_ = 2.0;
   Z(1,2).d_ = 2.0;
   Z(2,0).d_ = 2.0;
   Z(2,1).d_ = 2.0;
   Z(2,2).d_ = 2.0;

  matrix_fv output = stan::agrad::mdivide_left_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(1.0,output(0,0).val_);
  EXPECT_FLOAT_EQ(-4.3676905e-16,output(0,1).val_);
  EXPECT_FLOAT_EQ(-4.099285e-16,output(0,2).val_);
  EXPECT_FLOAT_EQ(-0.8,output(1,0).val_);
  EXPECT_FLOAT_EQ(0.6,output(1,1).val_);
  EXPECT_FLOAT_EQ(3.5868743e-16,output(1,2).val_);
  EXPECT_FLOAT_EQ(34.0 / 90.0,output(2,0).val_);
  EXPECT_FLOAT_EQ(2.0 / 90.0,output(2,1).val_);
  EXPECT_FLOAT_EQ(2.0 / 3.0,output(2,2).val_);
  EXPECT_FLOAT_EQ(5.655495e-16,output(0,0).d_);
  EXPECT_FLOAT_EQ(2.0,output(0,1).d_);
  EXPECT_FLOAT_EQ(2.0,output(0,2).d_);
  EXPECT_FLOAT_EQ(0.31999999,output(1,0).d_);
  EXPECT_FLOAT_EQ(-2.24,output(1,1).d_);
  EXPECT_FLOAT_EQ(-2.0,output(1,2).d_);
  EXPECT_FLOAT_EQ(-0.19061728,output(2,0).d_);
  EXPECT_FLOAT_EQ(0.51950616,output(2,1).d_);
  EXPECT_FLOAT_EQ(0.2962963,output(2,2).d_);
}
TEST(AgradFwdMatrix, mdivide_left_tri_low_matrix_fv_matrix) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_fv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  matrix_fv Z(3,3);
  Z << 1, 2, 3,
    6, 5, 4,
    7, 8, 9;
   Z(0,0).d_ = 2.0;
   Z(0,1).d_ = 2.0;
   Z(0,2).d_ = 2.0;
   Z(1,0).d_ = 2.0;
   Z(1,1).d_ = 2.0;
   Z(1,2).d_ = 2.0;
   Z(2,0).d_ = 2.0;
   Z(2,1).d_ = 2.0;
   Z(2,2).d_ = 2.0;

  matrix_fv output = stan::agrad::mdivide_left_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(1.0,output(0,0).val_);
  EXPECT_FLOAT_EQ(-4.3676905e-16,output(0,1).val_);
  EXPECT_FLOAT_EQ(-4.099285e-16,output(0,2).val_);
  EXPECT_FLOAT_EQ(-0.8,output(1,0).val_);
  EXPECT_FLOAT_EQ(0.6,output(1,1).val_);
  EXPECT_FLOAT_EQ(3.5868743e-16,output(1,2).val_);
  EXPECT_FLOAT_EQ(34.0 / 90.0,output(2,0).val_);
  EXPECT_FLOAT_EQ(2.0 / 90.0,output(2,1).val_);
  EXPECT_FLOAT_EQ(2.0 / 3.0,output(2,2).val_);
  EXPECT_FLOAT_EQ(-2.0,output(0,0).d_);
  EXPECT_FLOAT_EQ(9.9662511e-16,output(0,1).d_);
  EXPECT_FLOAT_EQ(9.1095228e-16,output(0,2).d_);
  EXPECT_FLOAT_EQ(2.3199999,output(1,0).d_);
  EXPECT_FLOAT_EQ(-0.23999999,output(1,1).d_);
  EXPECT_FLOAT_EQ(-1.0430403e-15,output(1,2).d_);
  EXPECT_FLOAT_EQ(-0.63506174,output(2,0).d_);
  EXPECT_FLOAT_EQ(0.075061731,output(2,1).d_);
  EXPECT_FLOAT_EQ(-0.14814815,output(2,2).d_);
}
TEST(AgradFwdMatrix, mdivide_left_tri_low_matrix_matrix_fv) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_fv;

  matrix_fv Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
   Y(2,2).d_ = 2.0;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    6, 5, 4,
    7, 8, 9;

  matrix_fv output = stan::agrad::mdivide_left_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(1.0,output(0,0).val_);
  EXPECT_FLOAT_EQ(-4.3676905e-16,output(0,1).val_);
  EXPECT_FLOAT_EQ(-4.099285e-16,output(0,2).val_);
  EXPECT_FLOAT_EQ(-0.8,output(1,0).val_);
  EXPECT_FLOAT_EQ(0.6,output(1,1).val_);
  EXPECT_FLOAT_EQ(3.5868743e-16,output(1,2).val_);
  EXPECT_FLOAT_EQ(34.0 / 90.0,output(2,0).val_);
  EXPECT_FLOAT_EQ(2.0 / 90.0,output(2,1).val_);
  EXPECT_FLOAT_EQ(2.0 / 3.0,output(2,2).val_);
  EXPECT_FLOAT_EQ(2.0,output(0,0).d_);
  EXPECT_FLOAT_EQ(2.0,output(0,1).d_);
  EXPECT_FLOAT_EQ(2.0,output(0,2).d_);
  EXPECT_FLOAT_EQ(-2.0,output(1,0).d_);
  EXPECT_FLOAT_EQ(-2.0,output(1,1).d_);
  EXPECT_FLOAT_EQ(-2.0,output(1,2).d_);
  EXPECT_FLOAT_EQ(4.0 / 9.0,output(2,0).d_);
  EXPECT_FLOAT_EQ(4.0 / 9.0,output(2,1).d_);
  EXPECT_FLOAT_EQ(4.0 / 9.0,output(2,2).d_);
}
TEST(AgradFwdMatrix, mdivide_left_tri_low_vector_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::math::vector_d;
  using stan::agrad::mdivide_left_tri_low;

  vector_fv fv1(4), fv2(3);
  vector_d v1(4), v2(3);
  matrix_fv fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_left_tri_low(fvm2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri_low(vm2,fv1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri_low(fvm2,v1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri_low(vm1,fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,v2), std::domain_error);
}
TEST(AgradFwdMatrix, mdivide_left_tri_low_matrix_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::mdivide_left_tri_low;

  matrix_fv fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_left_tri_low(fvm1,fvm2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,vm2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri_low(vm1,fvm2), std::domain_error);
}

