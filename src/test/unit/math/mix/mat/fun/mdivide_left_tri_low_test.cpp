#include <stan/math/fwd/mat/fun/mdivide_left_tri_low.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>

TEST(AgradMixMatrixMdivideLeftTriLow,fv_vector_fv_matrix_fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::vector_fv;

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

  matrix_fv output = stan::math::mdivide_left_tri_low(Y,Z);

  EXPECT_NEAR(1.0,output(0,0).val_.val(),1.0E-08);
  EXPECT_NEAR(0,output(1,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(-1.0 / 6.0,output(2,0).val_.val(),1.0E-08);
  EXPECT_NEAR(0,output(0,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(5.0 / 90.0,output(2,0).d_.val(),1.0E-08);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,fv_vector_fv_matrix_fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::vector_fv;

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

  matrix_fv output = stan::math::mdivide_left_tri_low(Y,Z);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_NEAR(2.0,h[0],1e-15);
  EXPECT_NEAR(0.0,h[1],1e-15);
  EXPECT_NEAR(0.0,h[2],1e-15);
  EXPECT_NEAR(0.0,h[3],1e-15);
  EXPECT_NEAR(0.0,h[4],1e-15);
  EXPECT_NEAR(0.0,h[5],1e-15);
}
TEST(AgradMixMatrixMdivideLeftTriLow,fv_vector_matrix_fv_1stDeriv) {
  using stan::math::matrix_fv;
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

  matrix_fv output = stan::math::mdivide_left_tri_low(Y,Z);

  EXPECT_NEAR(1.0,output(0,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(-1.0 / 6.0,output(2,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(0,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(4.0 / 6.0,output(1,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(1.0 / 2.0,output(2,0).d_.val(), 1.0E-08);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,fv_vector_matrix_fv_2ndDeriv) {
  using stan::math::matrix_fv;
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

  matrix_fv output = stan::math::mdivide_left_tri_low(Y,Z);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_NEAR(4.0,h[0],1e-15);
  EXPECT_NEAR(0.0,h[1],1e-15);
  EXPECT_NEAR(0.0,h[2],1e-15);
  EXPECT_NEAR(0.0,h[3],1e-15);
  EXPECT_NEAR(0.0,h[4],1e-15);
  EXPECT_NEAR(0.0,h[5],1e-15);
}
TEST(AgradMixMatrixMdivideLeftTriLow,fv_vector_fv_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  vector_fv Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 2.0;
   Z(1).d_ = 2.0;
   Z(2).d_ = 2.0;

  matrix_fv output = stan::math::mdivide_left_tri_low(Y,Z);

  EXPECT_NEAR(1.0,output(0,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(-1.0 / 6.0,output(2,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-4.0 / 6.0,output(1,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-4.0 / 9.0,output(2,0).d_.val(), 1.0E-08);

  AVEC q = createAVEC(Z(0).val(),Z(1).val(),Z(2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,fv_vector_fv_matrix_2ndDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  vector_fv Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 2.0;
   Z(1).d_ = 2.0;
   Z(2).d_ = 2.0;

  matrix_fv output = stan::math::mdivide_left_tri_low(Y,Z);

  EXPECT_NEAR(1.0,output(0,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(-1.0 / 6.0,output(2,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-4.0 / 6.0,output(1,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-4.0 / 9.0,output(2,0).d_.val(), 1.0E-08);

  AVEC q = createAVEC(Z(0).val(),Z(1).val(),Z(2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,fv_matrix_fv_matrix_fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;

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

  matrix_fv output = stan::math::mdivide_left_tri_low(Z,Y);

  EXPECT_NEAR(1.0,output(0,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,1).val_.val(),1.0E-08);
  EXPECT_NEAR(0,output(0,2).val_.val(), 1.0E-08);
  EXPECT_NEAR(-0.8,output(1,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0.6,output(1,1).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,2).val_.val(), 1.0E-08);
  EXPECT_NEAR(34.0 / 90.0,output(2,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 90.0,output(2,1).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 3.0,output(2,2).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,2).d_.val(), 1.0E-08);
  EXPECT_NEAR(288.0 / 900.0,output(1,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-2.24,output(1,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,2).d_.val(), 1.0E-08);
  EXPECT_NEAR(-0.19061728,output(2,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(0.5195061728395064,output(2,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(0.2962963,output(2,2).d_.val(), 1.0E-08);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,fv_matrix_fv_matrix_fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;

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

  matrix_fv output = stan::math::mdivide_left_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_NEAR(-2.0,h[0],1e-15);
  EXPECT_NEAR(0.0,h[1],1e-15);
  EXPECT_NEAR(0.0,h[2],1e-15);
  EXPECT_NEAR(0.0,h[3],1e-15);
  EXPECT_NEAR(0.0,h[4],1e-15);
  EXPECT_NEAR(0.0,h[5],1e-15);
}
TEST(AgradMixMatrixMdivideLeftTriLow,fv_matrix_fv_matrix_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::row_vector_fv;

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

  matrix_fv output = stan::math::mdivide_left_tri_low(Z,Y);

  EXPECT_NEAR(1.0,output(0,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,1).val_.val(),1.0E-08);
  EXPECT_NEAR(0,output(0,2).val_.val(),1.0E-08);
  EXPECT_NEAR(-0.8,output(1,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0.6,output(1,1).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,2).val_.val(),1.0E-08);
  EXPECT_NEAR(34.0 / 90.0,output(2,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 90.0,output(2,1).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 3.0,output(2,2).val_.val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(0,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,1).d_.val(),1.0E-08);
  EXPECT_NEAR(0,output(0,2).d_.val(),1.0E-08);
  EXPECT_NEAR(2.3199999999999985,output(1,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-0.24,output(1,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,2).d_.val(),1.0E-08);
  EXPECT_NEAR(-0.63506172839506148,output(2,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(0.075061731,output(2,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(-0.14814815,output(2,2).d_.val(), 1.0E-08);

  AVEC q = createAVEC(Z(0,0).val(),Z(0,1).val(),Z(0,2).val(),Z(1,0).val(),Z(1,1).val(),Z(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_NEAR(-1.0,h[0],1e-15);
  EXPECT_NEAR(0.0,h[1],1e-15);
  EXPECT_NEAR(0.0,h[2],1e-15);
  EXPECT_NEAR(0.0,h[3],1e-15);
  EXPECT_NEAR(0.0,h[4],1e-15);
  EXPECT_NEAR(0.0,h[5],1e-15);
}
TEST(AgradMixMatrixMdivideLeftTriLow,fv_matrix_fv_matrix_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::row_vector_fv;

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

  matrix_fv output = stan::math::mdivide_left_tri_low(Z,Y);

  AVEC q = createAVEC(Z(0,0).val(),Z(0,1).val(),Z(0,2).val(),Z(1,0).val(),Z(1,1).val(),Z(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_NEAR(0.0,h[3],1e-8);
  EXPECT_NEAR(0.0,h[4],1e-8);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,fv_matrix_matrix_fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::row_vector_fv;

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

  matrix_fv output = stan::math::mdivide_left_tri_low(Z,Y);

  EXPECT_NEAR(1.0,output(0,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,1).val_.val(),1.0E-08);
  EXPECT_NEAR(0,output(0,2).val_.val(),1.0E-08);
  EXPECT_NEAR(-0.8,output(1,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0.6,output(1,1).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,2).val_.val(),1.0E-08);
  EXPECT_NEAR(34.0 / 90.0,output(2,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 90.0,output(2,1).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 3.0,output(2,2).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,2).d_.val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,2).d_.val(), 1.0E-08);
  EXPECT_NEAR(4.0 / 9.0,output(2,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(4.0 / 9.0,output(2,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(4.0 / 9.0,output(2,2).d_.val(), 1.0E-08);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,fv_matrix_matrix_fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::row_vector_fv;

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

  matrix_fv output = stan::math::mdivide_left_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,fv_vector_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri_low;

  vector_fv fv1(4), fv2(3);
  vector_d v1(4), v2(3);
  matrix_fv fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_left_tri_low(fvm2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(vm2,fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm2,v1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(vm1,fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,v2), std::invalid_argument);
}
TEST(AgradMixMatrixMdivideLeftTriLow,fv_matrix_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::mdivide_left_tri_low;

  matrix_fv fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_left_tri_low(fvm1,fvm2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,vm2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(vm1,fvm2), std::invalid_argument);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_vector_ffv_matrix_ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  matrix_ffv Y(3,3);
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

   vector_ffv Z(3);
   Z << 1, 2, 3;
    Z(0).d_ = 2.0;
    Z(1).d_ = 2.0;
    Z(2).d_ = 2.0;

  matrix_ffv output = stan::math::mdivide_left_tri_low(Y,Z);

  EXPECT_NEAR(1.0,output(0,0).val_.val().val(),1.0E-08);
  EXPECT_NEAR(0,output(1,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(-1.0 / 6.0,output(2,0).val_.val().val(),1.0E-08);
  EXPECT_NEAR(0,output(0,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(5.0 / 90.0,output(2,0).d_.val().val(),1.0E-08);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_vector_ffv_matrix_ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  matrix_ffv Y(3,3);
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

   vector_ffv Z(3);
   Z << 1, 2, 3;
    Z(0).d_ = 2.0;
    Z(1).d_ = 2.0;
    Z(2).d_ = 2.0;

  matrix_ffv output = stan::math::mdivide_left_tri_low(Y,Z);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_vector_ffv_matrix_ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  matrix_ffv Y(3,3);
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

   vector_ffv Z(3);
   Z << 1, 2, 3;
    Z(0).d_ = 2.0;
    Z(1).d_ = 2.0;
    Z(2).d_ = 2.0;

  matrix_ffv output = stan::math::mdivide_left_tri_low(Y,Z);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_NEAR(2.0,h[0],1e-15);
  EXPECT_NEAR(0.0,h[1],1e-15);
  EXPECT_NEAR(0.0,h[2],1e-15);
  EXPECT_NEAR(0.0,h[3],1e-15);
  EXPECT_NEAR(0.0,h[4],1e-15);
  EXPECT_NEAR(0.0,h[5],1e-15);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_vector_ffv_matrix_ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  matrix_ffv Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 1.0;
   Y(0,1).d_ = 1.0;
   Y(0,2).d_ = 1.0;
   Y(1,0).d_ = 1.0;
   Y(1,1).d_ = 1.0;
   Y(1,2).d_ = 1.0;
   Y(2,0).d_ = 1.0;
   Y(2,1).d_ = 1.0;
   Y(2,2).d_ = 1.0;
   Y(0,0).val_.d_ = 1.0;
   Y(0,1).val_.d_ = 1.0;
   Y(0,2).val_.d_ = 1.0;
   Y(1,0).val_.d_ = 1.0;
   Y(1,1).val_.d_ = 1.0;
   Y(1,2).val_.d_ = 1.0;
   Y(2,0).val_.d_ = 1.0;
   Y(2,1).val_.d_ = 1.0;
   Y(2,2).val_.d_ = 1.0;

   vector_ffv Z(3);
   Z << 1, 2, 3;
    Z(0).d_ = 1.0;
    Z(1).d_ = 1.0;
    Z(2).d_ = 1.0;
    Z(0).val_.d_ = 1.0;
    Z(1).val_.d_ = 1.0;
    Z(2).val_.d_ = 1.0;


  matrix_ffv output = stan::math::mdivide_left_tri_low(Y,Z);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_NEAR(-2.0,h[0],1e-8);
  EXPECT_NEAR(0.0,h[1],1e-8);
  EXPECT_NEAR(0.0,h[2],1e-8);
  EXPECT_NEAR(0.0,h[3],1e-8);
  EXPECT_NEAR(0.0,h[4],1e-8);
  EXPECT_NEAR(0.0,h[5],1e-8);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_vector_matrix_ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::vector_d;

  matrix_ffv Y(3,3);
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

  matrix_ffv output = stan::math::mdivide_left_tri_low(Y,Z);

  EXPECT_NEAR(1.0,output(0,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(-1.0 / 6.0,output(2,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(0,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(4.0 / 6.0,output(1,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(1.0 / 2.0,output(2,0).d_.val().val(), 1.0E-08);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_vector_matrix_ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::vector_d;

  matrix_ffv Y(3,3);
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

  matrix_ffv output = stan::math::mdivide_left_tri_low(Y,Z);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_vector_matrix_ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::vector_d;

  matrix_ffv Y(3,3);
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

  matrix_ffv output = stan::math::mdivide_left_tri_low(Y,Z);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_NEAR(4.0,h[0],1e-15);
  EXPECT_NEAR(0.0,h[1],1e-15);
  EXPECT_NEAR(0.0,h[2],1e-15);
  EXPECT_NEAR(0.0,h[3],1e-15);
  EXPECT_NEAR(0.0,h[4],1e-15);
  EXPECT_NEAR(0.0,h[5],1e-15);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_vector_matrix_ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::vector_d;

  matrix_ffv Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 1.0;
   Y(0,1).d_ = 1.0;
   Y(0,2).d_ = 1.0;
   Y(1,0).d_ = 1.0;
   Y(1,1).d_ = 1.0;
   Y(1,2).d_ = 1.0;
   Y(2,0).d_ = 1.0;
   Y(2,1).d_ = 1.0;
   Y(2,2).d_ = 1.0;
   Y(0,0).val_.d_ = 1.0;
   Y(0,1).val_.d_ = 1.0;
   Y(0,2).val_.d_ = 1.0;
   Y(1,0).val_.d_ = 1.0;
   Y(1,1).val_.d_ = 1.0;
   Y(1,2).val_.d_ = 1.0;
   Y(2,0).val_.d_ = 1.0;
   Y(2,1).val_.d_ = 1.0;
   Y(2,2).val_.d_ = 1.0;

   vector_d Z(3);
   Z << 1, 2, 3;

  matrix_ffv output = stan::math::mdivide_left_tri_low(Y,Z);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_NEAR(-6.0,h[0],1e-8);
  EXPECT_NEAR(0.0,h[1],1e-8);
  EXPECT_NEAR(0.0,h[2],1e-8);
  EXPECT_NEAR(0.0,h[3],1e-8);
  EXPECT_NEAR(0.0,h[4],1e-8);
  EXPECT_NEAR(0.0,h[5],1e-8);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_vector_ffv_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  vector_ffv Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 2.0;
   Z(1).d_ = 2.0;
   Z(2).d_ = 2.0;

  matrix_ffv output = stan::math::mdivide_left_tri_low(Y,Z);

  EXPECT_NEAR(1.0,output(0,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(-1.0 / 6.0,output(2,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(-4.0 / 6.0,output(1,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(-4.0 / 9.0,output(2,0).d_.val().val(), 1.0E-08);

  AVEC q = createAVEC(Z(0).val().val(),Z(1).val().val(),Z(2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_vector_ffv_matrix_2ndDeriv_1) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  vector_ffv Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 2.0;
   Z(1).d_ = 2.0;
   Z(2).d_ = 2.0;

  matrix_ffv output = stan::math::mdivide_left_tri_low(Y,Z);

  AVEC q = createAVEC(Z(0).val().val(),Z(1).val().val(),Z(2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_vector_ffv_matrix_2ndDeriv_2) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  vector_ffv Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 2.0;
   Z(1).d_ = 2.0;
   Z(2).d_ = 2.0;

  matrix_ffv output = stan::math::mdivide_left_tri_low(Y,Z);

  AVEC q = createAVEC(Z(0).val().val(),Z(1).val().val(),Z(2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_vector_ffv_matrix_3rdDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  vector_ffv Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 1.0;
   Z(1).d_ = 1.0;
   Z(2).d_ = 1.0;
   Z(0).val_.d_ = 1.0;
   Z(1).val_.d_ = 1.0;
   Z(2).val_.d_ = 1.0;

  matrix_ffv output = stan::math::mdivide_left_tri_low(Y,Z);

  AVEC q = createAVEC(Z(0).val().val(),Z(1).val().val(),Z(2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_NEAR(0.0,h[0],1e-8);
  EXPECT_NEAR(0.0,h[1],1e-8);
  EXPECT_NEAR(0.0,h[2],1e-8);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_matrix_ffv_matrix_ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;

  matrix_ffv Y(3,3);
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

  matrix_ffv Z(3,3);
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

  matrix_ffv output = stan::math::mdivide_left_tri_low(Z,Y);

  EXPECT_NEAR(1.0,output(0,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,1).val_.val().val(),1.0E-08);
  EXPECT_NEAR(0,output(0,2).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(-0.8,output(1,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(0.6,output(1,1).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,2).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(34.0 / 90.0,output(2,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 90.0,output(2,1).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 3.0,output(2,2).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,1).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,2).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(288.0 / 900.0,output(1,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(-2.24,output(1,1).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,2).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(-0.19061728,output(2,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(0.5195061728395064,output(2,1).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(0.2962963,output(2,2).d_.val().val(), 1.0E-08);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_matrix_ffv_matrix_ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;

  matrix_ffv Y(3,3);
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

  matrix_ffv Z(3,3);
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

  matrix_ffv output = stan::math::mdivide_left_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_matrix_ffv_matrix_ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;

  matrix_ffv Y(3,3);
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

  matrix_ffv Z(3,3);
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

  matrix_ffv output = stan::math::mdivide_left_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_NEAR(-2.0,h[0],1e-15);
  EXPECT_NEAR(0.0,h[1],1e-15);
  EXPECT_NEAR(0.0,h[2],1e-15);
  EXPECT_NEAR(0.0,h[3],1e-15);
  EXPECT_NEAR(0.0,h[4],1e-15);
  EXPECT_NEAR(0.0,h[5],1e-15);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_matrix_ffv_matrix_ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;

  matrix_ffv Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 1.0;
   Y(0,1).d_ = 1.0;
   Y(0,2).d_ = 1.0;
   Y(1,0).d_ = 1.0;
   Y(1,1).d_ = 1.0;
   Y(1,2).d_ = 1.0;
   Y(2,0).d_ = 1.0;
   Y(2,1).d_ = 1.0;
   Y(2,2).d_ = 1.0;
   Y(0,0).val_.d_ = 1.0;
   Y(0,1).val_.d_ = 1.0;
   Y(0,2).val_.d_ = 1.0;
   Y(1,0).val_.d_ = 1.0;
   Y(1,1).val_.d_ = 1.0;
   Y(1,2).val_.d_ = 1.0;
   Y(2,0).val_.d_ = 1.0;
   Y(2,1).val_.d_ = 1.0;
   Y(2,2).val_.d_ = 1.0;

  matrix_ffv Z(3,3);
  Z << 1, 2, 3,
    6, 5, 4,
    7, 8, 9;
   Z(0,0).d_ = 1.0;
   Z(0,1).d_ = 1.0;
   Z(0,2).d_ = 1.0;
   Z(1,0).d_ = 1.0;
   Z(1,1).d_ = 1.0;
   Z(1,2).d_ = 1.0;
   Z(2,0).d_ = 1.0;
   Z(2,1).d_ = 1.0;
   Z(2,2).d_ = 1.0;
   Z(0,0).val_.d_ = 1.0;
   Z(0,1).val_.d_ = 1.0;
   Z(0,2).val_.d_ = 1.0;
   Z(1,0).val_.d_ = 1.0;
   Z(1,1).val_.d_ = 1.0;
   Z(1,2).val_.d_ = 1.0;
   Z(2,0).val_.d_ = 1.0;
   Z(2,1).val_.d_ = 1.0;
   Z(2,2).val_.d_ = 1.0;

  matrix_ffv output = stan::math::mdivide_left_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_NEAR(2.0,h[0],1e-8);
  EXPECT_NEAR(0.0,h[1],1e-8);
  EXPECT_NEAR(0.0,h[2],1e-8);
  EXPECT_NEAR(0.0,h[3],1e-8);
  EXPECT_NEAR(0.0,h[4],1e-8);
  EXPECT_NEAR(0.0,h[5],1e-8);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_matrix_ffv_matrix_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  matrix_ffv Z(3,3);
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

  matrix_ffv output = stan::math::mdivide_left_tri_low(Z,Y);

  EXPECT_NEAR(1.0,output(0,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,1).val_.val().val(),1.0E-08);
  EXPECT_NEAR(0,output(0,2).val_.val().val(),1.0E-08);
  EXPECT_NEAR(-0.8,output(1,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(0.6,output(1,1).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,2).val_.val().val(),1.0E-08);
  EXPECT_NEAR(34.0 / 90.0,output(2,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 90.0,output(2,1).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 3.0,output(2,2).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(0,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,1).d_.val().val(),1.0E-08);
  EXPECT_NEAR(0,output(0,2).d_.val().val(),1.0E-08);
  EXPECT_NEAR(2.3199999999999985,output(1,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(-0.24,output(1,1).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,2).d_.val().val(),1.0E-08);
  EXPECT_NEAR(-0.63506172839506148,output(2,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(0.075061731,output(2,1).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(-0.14814815,output(2,2).d_.val().val(), 1.0E-08);

  AVEC q = createAVEC(Z(0,0).val().val(),Z(0,1).val().val(),Z(0,2).val().val(),Z(1,0).val().val(),Z(1,1).val().val(),Z(1,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_NEAR(-1.0,h[0],1e-15);
  EXPECT_NEAR(0.0,h[1],1e-15);
  EXPECT_NEAR(0.0,h[2],1e-15);
  EXPECT_NEAR(0.0,h[3],1e-15);
  EXPECT_NEAR(0.0,h[4],1e-15);
  EXPECT_NEAR(0.0,h[5],1e-15);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_matrix_ffv_matrix_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  matrix_ffv Z(3,3);
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

  matrix_ffv output = stan::math::mdivide_left_tri_low(Z,Y);

  AVEC q = createAVEC(Z(0,0).val().val(),Z(0,1).val().val(),Z(0,2).val().val(),Z(1,0).val().val(),Z(1,1).val().val(),Z(1,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_NEAR(0.0,h[3],1e-8);
  EXPECT_NEAR(0.0,h[4],1e-8);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_matrix_ffv_matrix_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  matrix_ffv Z(3,3);
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

  matrix_ffv output = stan::math::mdivide_left_tri_low(Z,Y);

  AVEC q = createAVEC(Z(0,0).val().val(),Z(0,1).val().val(),Z(0,2).val().val(),Z(1,0).val().val(),Z(1,1).val().val(),Z(1,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_NEAR(0.0,h[3],1e-8);
  EXPECT_NEAR(0.0,h[4],1e-8);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_matrix_ffv_matrix_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  matrix_ffv Z(3,3);
  Z << 1, 2, 3,
    6, 5, 4,
    7, 8, 9;
   Z(0,0).d_ = 1.0;
   Z(0,1).d_ = 1.0;
   Z(0,2).d_ = 1.0;
   Z(1,0).d_ = 1.0;
   Z(1,1).d_ = 1.0;
   Z(1,2).d_ = 1.0;
   Z(2,0).d_ = 1.0;
   Z(2,1).d_ = 1.0;
   Z(2,2).d_ = 1.0;
   Z(0,0).val_.d_ = 1.0;
   Z(0,1).val_.d_ = 1.0;
   Z(0,2).val_.d_ = 1.0;
   Z(1,0).val_.d_ = 1.0;
   Z(1,1).val_.d_ = 1.0;
   Z(1,2).val_.d_ = 1.0;
   Z(2,0).val_.d_ = 1.0;
   Z(2,1).val_.d_ = 1.0;
   Z(2,2).val_.d_ = 1.0;

  matrix_ffv output = stan::math::mdivide_left_tri_low(Z,Y);

  AVEC q = createAVEC(Z(0,0).val().val(),Z(0,1).val().val(),Z(0,2).val().val(),Z(1,0).val().val(),Z(1,1).val().val(),Z(1,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_NEAR(-6.0,h[0],1e-8);
  EXPECT_NEAR(0.0,h[1],1e-8);
  EXPECT_NEAR(0.0,h[2],1e-8);
  EXPECT_NEAR(0.0,h[3],1e-8);
  EXPECT_NEAR(0.0,h[4],1e-8);
  EXPECT_NEAR(0.0,h[5],1e-8);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_matrix_matrix_ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;

  matrix_ffv Y(3,3);
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

  matrix_ffv output = stan::math::mdivide_left_tri_low(Z,Y);

  EXPECT_NEAR(1.0,output(0,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,1).val_.val().val(),1.0E-08);
  EXPECT_NEAR(0,output(0,2).val_.val().val(),1.0E-08);
  EXPECT_NEAR(-0.8,output(1,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(0.6,output(1,1).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,2).val_.val().val(),1.0E-08);
  EXPECT_NEAR(34.0 / 90.0,output(2,0).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 90.0,output(2,1).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 3.0,output(2,2).val_.val().val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,1).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,2).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,1).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,2).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(4.0 / 9.0,output(2,0).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(4.0 / 9.0,output(2,1).d_.val().val(), 1.0E-08);
  EXPECT_NEAR(4.0 / 9.0,output(2,2).d_.val().val(), 1.0E-08);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_matrix_matrix_ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;

  matrix_ffv Y(3,3);
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

  matrix_ffv output = stan::math::mdivide_left_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_matrix_matrix_ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;

  matrix_ffv Y(3,3);
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

  matrix_ffv output = stan::math::mdivide_left_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_matrix_matrix_ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;

  matrix_ffv Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 1.0;
   Y(0,1).d_ = 1.0;
   Y(0,2).d_ = 1.0;
   Y(1,0).d_ = 1.0;
   Y(1,1).d_ = 1.0;
   Y(1,2).d_ = 1.0;
   Y(2,0).d_ = 1.0;
   Y(2,1).d_ = 1.0;
   Y(2,2).d_ = 1.0;
   Y(0,0).val_.d_ = 1.0;
   Y(0,1).val_.d_ = 1.0;
   Y(0,2).val_.d_ = 1.0;
   Y(1,0).val_.d_ = 1.0;
   Y(1,1).val_.d_ = 1.0;
   Y(1,2).val_.d_ = 1.0;
   Y(2,0).val_.d_ = 1.0;
   Y(2,1).val_.d_ = 1.0;
   Y(2,2).val_.d_ = 1.0;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    6, 5, 4,
    7, 8, 9;

  matrix_ffv output = stan::math::mdivide_left_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_NEAR(0.0,h[0],1e-8);
  EXPECT_NEAR(0.0,h[1],1e-8);
  EXPECT_NEAR(0.0,h[2],1e-8);
  EXPECT_NEAR(0.0,h[3],1e-8);
  EXPECT_NEAR(0.0,h[4],1e-8);
  EXPECT_NEAR(0.0,h[5],1e-8);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_vector_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri_low;

  vector_ffv fv1(4), fv2(3);
  vector_d v1(4), v2(3);
  matrix_ffv fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_left_tri_low(fvm2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(vm2,fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm2,v1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(vm1,fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,v2), std::invalid_argument);
}
TEST(AgradMixMatrixMdivideLeftTriLow,ffv_matrix_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::mdivide_left_tri_low;

  matrix_ffv fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_left_tri_low(fvm1,fvm2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,vm2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(vm1,fvm2), std::invalid_argument);
}
