#include <stan/math/fwd/mat/fun/mdivide_right_tri_low.hpp>
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

using stan::math::fvar;
TEST(AgradMixMatrixMdivideRightTriLow,fv__rowvector_fv_matrix_fv_1stDeriv) {
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

   row_vector_fv Z(3);
   Z << 1, 2, 3;
    Z(0).d_ = 2.0;
    Z(1).d_ = 2.0;
    Z(2).d_ = 2.0;

  matrix_fv output = stan::math::mdivide_right_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(-2.0 / 3.0,output(0,0).val_.val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.5,output(0,2).val_.val());
  EXPECT_FLOAT_EQ(5.0 / 3.0,output(0,0).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(0,2).d_.val());

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(2.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(1.0/6.0,h[3]);
  EXPECT_FLOAT_EQ(-1.0/9.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,fv__rowvector_fv_matrix_fv_2ndDeriv) {
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

   row_vector_fv Z(3);
   Z << 1, 2, 3;
    Z(0).d_ = 2.0;
    Z(1).d_ = 2.0;
    Z(2).d_ = 2.0;

  matrix_fv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-3.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(-0.5,h[3]);
  EXPECT_FLOAT_EQ(8.0/27.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,fv__rowvector_matrix_fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::row_vector_d;

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

   row_vector_d Z(3);
   Z << 1, 2, 3;

  matrix_fv output = stan::math::mdivide_right_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(-2.0 / 3.0,output(0,0).val_.val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.5,output(0,2).val_.val());
  EXPECT_FLOAT_EQ(11.0 / 9.0,output(0,0).d_.val());
  EXPECT_FLOAT_EQ(5.0 / 90.0,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,2).d_.val());

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(2.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(1.0/6.0,h[3]);
  EXPECT_FLOAT_EQ(-1.0/9.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,fv__rowvector_matrix_fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::row_vector_d;

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

   row_vector_d Z(3);
   Z << 1, 2, 3;

  matrix_fv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-23.0/9.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(-7.0/18.0,h[3]);
  EXPECT_FLOAT_EQ(2.0/9.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,fv__rowvector_fv_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  row_vector_fv Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 2.0;
   Z(1).d_ = 2.0;
   Z(2).d_ = 2.0;

  matrix_fv output = stan::math::mdivide_right_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(-2.0 / 3.0,output(0,0).val_.val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.5,output(0,2).val_.val());
  EXPECT_FLOAT_EQ(4.0 / 9.0,output(0,0).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 9.0,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(3.0 / 9.0,output(0,2).d_.val());

  AVEC q = createAVEC(Z(0).val(),Z(1).val(),Z(2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(-2.0/3.0,h[1]);
  EXPECT_FLOAT_EQ(-1.0/9.0,h[2]);
}
TEST(AgradMixMatrixMdivideRightTriLow,fv__rowvector_fv_matrix_2ndDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  row_vector_fv Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 2.0;
   Z(1).d_ = 2.0;
   Z(2).d_ = 2.0;

  matrix_fv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Z(0).val(),Z(1).val(),Z(2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixMdivideRightTriLow,fv__matrix_fv_matrix_fv_1stDeriv) {
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

  matrix_fv output = stan::math::mdivide_right_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(-2.0 / 3.0,output(0,0).val_.val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.5,output(0,2).val_.val());
  EXPECT_FLOAT_EQ(20.0 / 9.0,output(1,0).val_.val());
  EXPECT_FLOAT_EQ(5.0 / 9.0,output(1,1).val_.val());
  EXPECT_FLOAT_EQ(6.0 / 9.0,output(1,2).val_.val());
  EXPECT_FLOAT_EQ(6.0 / 9.0,output(2,0).val_.val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(2,1).val_.val());
  EXPECT_FLOAT_EQ(1.5,output(2,2).val_.val());
  EXPECT_FLOAT_EQ(15.0 / 9.0,output(0,0).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(0,2).d_.val());
  EXPECT_FLOAT_EQ(-14.0 / 3.0,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(-3.0 / 9.0,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 9.0,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(-5.0 / 3.0,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(2,2).d_.val());

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(2.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(1.0/6.0,h[3]);
  EXPECT_FLOAT_EQ(-1.0/9.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,fv__matrix_fv_matrix_fv_2ndDeriv) {
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

  matrix_fv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-3.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(-0.5,h[3]);
  EXPECT_FLOAT_EQ(8.0/27.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,fv__matrix_fv_matrix_1stDeriv) {
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

  matrix_fv output = stan::math::mdivide_right_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(-2.0 / 3.0,output(0,0).val_.val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.5,output(0,2).val_.val());
  EXPECT_FLOAT_EQ(20.0 / 9.0,output(1,0).val_.val());
  EXPECT_FLOAT_EQ(5.0 / 9.0,output(1,1).val_.val());
  EXPECT_FLOAT_EQ(6.0 / 9.0,output(1,2).val_.val());
  EXPECT_FLOAT_EQ(6.0 / 9.0,output(2,0).val_.val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(2,1).val_.val());
  EXPECT_FLOAT_EQ(1.5,output(2,2).val_.val());
  EXPECT_FLOAT_EQ(4.0 / 9.0,output(0,0).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 9.0,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(2.0 / 6.0,output(0,2).d_.val());
  EXPECT_FLOAT_EQ(4.0 / 9.0,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 9.0,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(3.0 / 9.0,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(4.0 / 9.0,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 9.0,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(2.0 / 6.0,output(2,2).d_.val());

  AVEC q = createAVEC(Z(0,0).val(),Z(0,1).val(),Z(0,2).val(),Z(1,0).val(),Z(1,1).val(),Z(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(-2.0/3.0,h[1]);
  EXPECT_FLOAT_EQ(-1.0/9.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,fv__matrix_fv_matrix_2ndDeriv) {
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

  matrix_fv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Z(0,0).val(),Z(0,1).val(),Z(0,2).val(),Z(1,0).val(),Z(1,1).val(),Z(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,fv__matrix_matrix_fv_1stDeriv) {
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

  matrix_fv output = stan::math::mdivide_right_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(-2.0 / 3.0,output(0,0).val_.val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.5,output(0,2).val_.val());
  EXPECT_FLOAT_EQ(20.0 / 9.0,output(1,0).val_.val());
  EXPECT_FLOAT_EQ(5.0 / 9.0,output(1,1).val_.val());
  EXPECT_FLOAT_EQ(6.0 / 9.0,output(1,2).val_.val());
  EXPECT_FLOAT_EQ(6.0 / 9.0,output(2,0).val_.val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(2,1).val_.val());
  EXPECT_FLOAT_EQ(1.5,output(2,2).val_.val());
  EXPECT_FLOAT_EQ(11.0 / 9.0,output(0,0).d_.val());
  EXPECT_FLOAT_EQ(5.0 / 90.0,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,2).d_.val());
  EXPECT_FLOAT_EQ(-46.0 / 9.0,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(-4.0 / 9.0,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(-2.0 / 9.0,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(-19.0 / 9.0,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(-25.0 / 90.0,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(-3.0 / 6.0,output(2,2).d_.val());

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(2.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(1.0/6.0,h[3]);
  EXPECT_FLOAT_EQ(-1.0/9.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,fv__matrix_matrix_fv_2ndDeriv) {
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

  matrix_fv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-23.0/9.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(-7.0/18.0,h[3]);
  EXPECT_FLOAT_EQ(2.0/9.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,fv__rowvector_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_tri_low;

  row_vector_fv fv1(4), fv2(3);
  row_vector_d v1(4), v2(3);
  matrix_fv fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_right_tri_low(fv1,fvm2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(fv1,vm2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(v1,fvm2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(fv2,fvm1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(fv2,vm1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(v2,fvm1), std::invalid_argument);
}
TEST(AgradMixMatrixMdivideRightTriLow,fv__matrix_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::mdivide_right_tri_low;

  matrix_fv fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_right_tri_low(fvm1,fvm2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(fvm1,vm2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(vm1,fvm2), std::invalid_argument);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__rowvector_ffv_matrix_ffv_1stDeriv) {
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

   row_vector_ffv Z(3);
   Z << 1, 2, 3;
    Z(0).d_ = 2.0;
    Z(1).d_ = 2.0;
    Z(2).d_ = 2.0;

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(-2.0 / 3.0,output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(0.5,output(0,2).val_.val().val());
  EXPECT_FLOAT_EQ(5.0 / 3.0,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(0,2).d_.val().val());

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(1.0/6.0,h[3]);
  EXPECT_FLOAT_EQ(-1.0/9.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__rowvector_ffv_matrix_ffv_2ndDeriv_1) {
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

   row_vector_ffv Z(3);
   Z << 1, 2, 3;
    Z(0).d_ = 2.0;
    Z(1).d_ = 2.0;
    Z(2).d_ = 2.0;

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__rowvector_ffv_matrix_ffv_2ndDeriv_2) {
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

   row_vector_ffv Z(3);
   Z << 1, 2, 3;
    Z(0).d_ = 2.0;
    Z(1).d_ = 2.0;
    Z(2).d_ = 2.0;

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-3.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(-0.5,h[3]);
  EXPECT_FLOAT_EQ(8.0/27.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__rowvector_ffv_matrix_ffv_3rdDeriv) {
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

   row_vector_ffv Z(3);
   Z << 1, 2, 3;
    Z(0).d_ = 1.0;
    Z(1).d_ = 1.0;
    Z(2).d_ = 1.0;
    Z(0).val_.d_ = 1.0;
    Z(1).val_.d_ = 1.0;
    Z(2).val_.d_ = 1.0;

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(4.7592592,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.56481481,h[3]);
  EXPECT_FLOAT_EQ(-0.30864197,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__rowvector_matrix_ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_d;

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

   row_vector_d Z(3);
   Z << 1, 2, 3;

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(-2.0 / 3.0,output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(0.5,output(0,2).val_.val().val());
  EXPECT_FLOAT_EQ(11.0 / 9.0,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(5.0 / 90.0,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,2).d_.val().val());

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(1.0/6.0,h[3]);
  EXPECT_FLOAT_EQ(-1.0/9.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__rowvector_matrix_ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_d;

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

   row_vector_d Z(3);
   Z << 1, 2, 3;

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__rowvector_matrix_ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_d;

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

   row_vector_d Z(3);
   Z << 1, 2, 3;

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-23.0/9.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(-7.0/18.0,h[3]);
  EXPECT_FLOAT_EQ(2.0/9.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__rowvector_matrix_ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_d;

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

   row_vector_d Z(3);
   Z << 1, 2, 3;

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(3.7592592,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.39814815,h[3]);
  EXPECT_FLOAT_EQ(-0.20987654,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__rowvector_ffv_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  row_vector_ffv Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 2.0;
   Z(1).d_ = 2.0;
   Z(2).d_ = 2.0;

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(-2.0 / 3.0,output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(0.5,output(0,2).val_.val().val());
  EXPECT_FLOAT_EQ(4.0 / 9.0,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 9.0,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(3.0 / 9.0,output(0,2).d_.val().val());

  AVEC q = createAVEC(Z(0).val().val(),Z(1).val().val(),Z(2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(-2.0/3.0,h[1]);
  EXPECT_FLOAT_EQ(-1.0/9.0,h[2]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__rowvector_ffv_matrix_2ndDeriv_1) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  row_vector_ffv Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 2.0;
   Z(1).d_ = 2.0;
   Z(2).d_ = 2.0;

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Z(0).val().val(),Z(1).val().val(),Z(2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__rowvector_ffv_matrix_2ndDeriv_2) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  row_vector_ffv Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 2.0;
   Z(1).d_ = 2.0;
   Z(2).d_ = 2.0;

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Z(0).val().val(),Z(1).val().val(),Z(2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__rowvector_ffv_matrix_3rdDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  row_vector_ffv Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 1.0;
   Z(1).d_ = 1.0;
   Z(2).d_ = 1.0;
   Z(0).val_.d_ = 1.0;
   Z(1).val_.d_ = 1.0;
   Z(2).val_.d_ = 1.0;

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Z(0).val().val(),Z(1).val().val(),Z(2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__matrix_ffv_matrix_ffv_1stDeriv) {
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

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(-2.0 / 3.0,output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(0.5,output(0,2).val_.val().val());
  EXPECT_FLOAT_EQ(20.0 / 9.0,output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(5.0 / 9.0,output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(6.0 / 9.0,output(1,2).val_.val().val());
  EXPECT_FLOAT_EQ(6.0 / 9.0,output(2,0).val_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(2,1).val_.val().val());
  EXPECT_FLOAT_EQ(1.5,output(2,2).val_.val().val());
  EXPECT_FLOAT_EQ(15.0 / 9.0,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ(-14.0 / 3.0,output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(-3.0 / 9.0,output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 9.0,output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ(-5.0 / 3.0,output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(2,2).d_.val().val());

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(1.0/6.0,h[3]);
  EXPECT_FLOAT_EQ(-1.0/9.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__matrix_ffv_matrix_ffv_2ndDeriv_1) {
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

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__matrix_ffv_matrix_ffv_2ndDeriv_2) {
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

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-3.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(-0.5,h[3]);
  EXPECT_FLOAT_EQ(8.0/27.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__matrix_ffv_matrix_ffv_3rdDeriv) {
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

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(4.7592592,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.56481481,h[3]);
  EXPECT_FLOAT_EQ(-0.30864197,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__matrix_ffv_matrix_1stDeriv) {
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

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(-2.0 / 3.0,output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(0.5,output(0,2).val_.val().val());
  EXPECT_FLOAT_EQ(20.0 / 9.0,output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(5.0 / 9.0,output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(6.0 / 9.0,output(1,2).val_.val().val());
  EXPECT_FLOAT_EQ(6.0 / 9.0,output(2,0).val_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(2,1).val_.val().val());
  EXPECT_FLOAT_EQ(1.5,output(2,2).val_.val().val());
  EXPECT_FLOAT_EQ(4.0 / 9.0,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 9.0,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(2.0 / 6.0,output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ(4.0 / 9.0,output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 9.0,output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(3.0 / 9.0,output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ(4.0 / 9.0,output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 9.0,output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ(2.0 / 6.0,output(2,2).d_.val().val());

  AVEC q = createAVEC(Z(0,0).val().val(),Z(0,1).val().val(),Z(0,2).val().val(),Z(1,0).val().val(),Z(1,1).val().val(),Z(1,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(-2.0/3.0,h[1]);
  EXPECT_FLOAT_EQ(-1.0/9.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__matrix_ffv_matrix_2ndDeriv_1) {
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

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Z(0,0).val().val(),Z(0,1).val().val(),Z(0,2).val().val(),Z(1,0).val().val(),Z(1,1).val().val(),Z(1,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__matrix_ffv_matrix_2ndDeriv_2) {
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

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Z(0,0).val().val(),Z(0,1).val().val(),Z(0,2).val().val(),Z(1,0).val().val(),Z(1,1).val().val(),Z(1,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__matrix_ffv_matrix_3rdDeriv) {
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

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Z(0,0).val().val(),Z(0,1).val().val(),Z(0,2).val().val(),Z(1,0).val().val(),Z(1,1).val().val(),Z(1,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__matrix_matrix_ffv_1stDeriv) {
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

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  EXPECT_FLOAT_EQ(-2.0 / 3.0,output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(0.5,output(0,2).val_.val().val());
  EXPECT_FLOAT_EQ(20.0 / 9.0,output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(5.0 / 9.0,output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(6.0 / 9.0,output(1,2).val_.val().val());
  EXPECT_FLOAT_EQ(6.0 / 9.0,output(2,0).val_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 6.0,output(2,1).val_.val().val());
  EXPECT_FLOAT_EQ(1.5,output(2,2).val_.val().val());
  EXPECT_FLOAT_EQ(11.0 / 9.0,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(5.0 / 90.0,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(-1.0 / 6.0,output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ(-46.0 / 9.0,output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(-4.0 / 9.0,output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(-2.0 / 9.0,output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ(-19.0 / 9.0,output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ(-25.0 / 90.0,output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ(-3.0 / 6.0,output(2,2).d_.val().val());

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(1.0/6.0,h[3]);
  EXPECT_FLOAT_EQ(-1.0/9.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__matrix_matrix_ffv_2ndDeriv_1) {
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

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__matrix_matrix_ffv_2ndDeriv_2) {
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

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-23.0/9.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(-7.0/18.0,h[3]);
  EXPECT_FLOAT_EQ(2.0/9.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__matrix_matrix_ffv_3rdDeriv) {
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

  matrix_ffv output = stan::math::mdivide_right_tri_low(Z,Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(3.7592592,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.39814815,h[3]);
  EXPECT_FLOAT_EQ(-0.20987654,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__rowvector_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_tri_low;

  row_vector_ffv fv1(4), fv2(3);
  row_vector_d v1(4), v2(3);
  matrix_ffv fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_right_tri_low(fv1,fvm2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(fv1,vm2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(v1,fvm2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(fv2,fvm1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(fv2,vm1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(v2,fvm1), std::invalid_argument);
}
TEST(AgradMixMatrixMdivideRightTriLow,ffv__matrix_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::mdivide_right_tri_low;

  matrix_ffv fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_right_tri_low(fvm1,fvm2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(fvm1,vm2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri_low(vm1,fvm2), std::invalid_argument);
}
