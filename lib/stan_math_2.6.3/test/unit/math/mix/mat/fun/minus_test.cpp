#include <stan/math/prim/mat/fun/minus.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

using stan::math::fvar;

TEST(AgradMixMatrixMinus, fv_scalar_1stDeriv) {
  using stan::math::minus;
  using stan::math::var;
  double x = 10;
  fvar<var> v = 11;
   v.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(-10, minus(x));
  EXPECT_FLOAT_EQ(-11, minus(v).val_.val());
  EXPECT_FLOAT_EQ( -1, minus(v).d_.val());

  AVEC q = createAVEC(v.val());
  VEC h;
  minus(v).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-1,h[0]);
}
TEST(AgradMixMatrixMinus, fv_scalar_2ndDeriv) {
  using stan::math::minus;
  using stan::math::var;
  double x = 10;
  fvar<var> v = 11;
   v.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(-10, minus(x));
  EXPECT_FLOAT_EQ(-11, minus(v).val_.val());
  EXPECT_FLOAT_EQ( -1, minus(v).d_.val());

  AVEC q = createAVEC(v.val());
  VEC h;
  minus(v).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
}
TEST(AgradMixMatrixMinus, fv_vector_1stDeriv) {
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::minus;

  vector_d d(3);
  vector_fv v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  
  vector_d output_d;
  output_d = minus(d);
  EXPECT_FLOAT_EQ(100, output_d[0]);
  EXPECT_FLOAT_EQ(0, output_d[1]);
  EXPECT_FLOAT_EQ(-1, output_d[2]);

  vector_fv output;
  output = minus(v);
  EXPECT_FLOAT_EQ(100, output[0].val_.val());
  EXPECT_FLOAT_EQ(0, output[1].val_.val());
  EXPECT_FLOAT_EQ(-1, output[2].val_.val());
  EXPECT_FLOAT_EQ(-1, output[0].d_.val());
  EXPECT_FLOAT_EQ(-1, output[1].d_.val());
  EXPECT_FLOAT_EQ(-1, output[2].d_.val());


  AVEC q = createAVEC(v(0).val(),v(1).val(),v(2).val());
  VEC h;
  output[0].val_.grad(q,h);
  EXPECT_FLOAT_EQ(-1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, fv_vector_2ndDeriv) {
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::minus;

  vector_d d(3);
  vector_fv v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  
  vector_d output_d;
  vector_fv output;
  output = minus(v);

  AVEC q = createAVEC(v(0).val(),v(1).val(),v(2).val());
  VEC h;
  output[0].d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, fv_rowvector_1stDeriv) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;
  using stan::math::minus;

  row_vector_d d(3);
  row_vector_fv v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  
  row_vector_d output_d;
  output_d = minus(d);
  EXPECT_FLOAT_EQ(100, output_d[0]);
  EXPECT_FLOAT_EQ(0, output_d[1]);
  EXPECT_FLOAT_EQ(-1, output_d[2]);

  row_vector_fv output;
  output = minus(v);
  EXPECT_FLOAT_EQ(100, output[0].val_.val());
  EXPECT_FLOAT_EQ(0, output[1].val_.val());
  EXPECT_FLOAT_EQ(-1, output[2].val_.val());
  EXPECT_FLOAT_EQ(-1, output[0].d_.val());
  EXPECT_FLOAT_EQ(-1, output[1].d_.val());
  EXPECT_FLOAT_EQ(-1, output[2].d_.val());

  AVEC q = createAVEC(v(0).val(),v(1).val(),v(2).val());
  VEC h;
  output[0].val_.grad(q,h);
  EXPECT_FLOAT_EQ(-1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, fv_rowvector_2ndDeriv) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;
  using stan::math::minus;

  row_vector_d d(3);
  row_vector_fv v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  
  row_vector_fv output;
  output = minus(v);

  AVEC q = createAVEC(v(0).val(),v(1).val(),v(2).val());
  VEC h;
  output[0].d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, fv_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::minus;

  matrix_d d(2, 3);
  matrix_fv v(2, 3);

  d << -100, 0, 1, 20, -40, 2;
  v << -100, 0, 1, 20, -40, 2;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(0,2).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
   v(1,2).d_ = 1.0;

  matrix_d output_d = minus(d);
  EXPECT_FLOAT_EQ(100, output_d(0,0));
  EXPECT_FLOAT_EQ(  0, output_d(0,1));
  EXPECT_FLOAT_EQ( -1, output_d(0,2));
  EXPECT_FLOAT_EQ(-20, output_d(1,0));
  EXPECT_FLOAT_EQ( 40, output_d(1,1));
  EXPECT_FLOAT_EQ( -2, output_d(1,2));

  matrix_fv output = minus(v);
  EXPECT_FLOAT_EQ(100, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(  0, output(0,1).val_.val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val_.val());
  EXPECT_FLOAT_EQ(-20, output(1,0).val_.val());
  EXPECT_FLOAT_EQ( 40, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( -2, output(1,2).val_.val());
  EXPECT_FLOAT_EQ( -1, output(0,0).d_.val());
  EXPECT_FLOAT_EQ( -1, output(0,1).d_.val());
  EXPECT_FLOAT_EQ( -1, output(0,2).d_.val());
  EXPECT_FLOAT_EQ( -1, output(1,0).d_.val());
  EXPECT_FLOAT_EQ( -1, output(1,1).d_.val());
  EXPECT_FLOAT_EQ( -1, output(1,2).d_.val());

  AVEC q = createAVEC(v(0,0).val(),v(0,1).val(),v(0,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, fv_matrix_2ndDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::minus;

  matrix_d d(2, 3);
  matrix_fv v(2, 3);

  d << -100, 0, 1, 20, -40, 2;
  v << -100, 0, 1, 20, -40, 2;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(0,2).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
   v(1,2).d_ = 1.0;

  matrix_fv output = minus(v);

  AVEC q = createAVEC(v(0,0).val(),v(0,1).val(),v(0,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, ffv_scalar_1stDeriv) {
  using stan::math::minus;
  using stan::math::var;
  double x = 10;
  fvar<fvar<var> > v = 11;
   v.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(-10, minus(x));
  EXPECT_FLOAT_EQ(-11, minus(v).val_.val().val());
  EXPECT_FLOAT_EQ( -1, minus(v).d_.val().val());

  AVEC q = createAVEC(v.val().val());
  VEC h;
  minus(v).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-1,h[0]);
}
TEST(AgradMixMatrixMinus, ffv_scalar_2ndDeriv_1) {
  using stan::math::minus;
  using stan::math::var;

  fvar<fvar<var> > v = 11;
   v.d_ = 1.0;
  
  AVEC q = createAVEC(v.val().val());
  VEC h;
  minus(v).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
}
TEST(AgradMixMatrixMinus, ffv_scalar_2ndDeriv_2) {
  using stan::math::minus;
  using stan::math::var;

  fvar<fvar<var> > v = 11;
   v.d_ = 1.0;

  AVEC q = createAVEC(v.val().val());
  VEC h;
  minus(v).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
}
TEST(AgradMixMatrixMinus, ffv_scalar_3rdDeriv) {
  using stan::math::minus;
  using stan::math::var;

  fvar<fvar<var> > v = 11;
   v.d_ = 1.0;
  
  AVEC q = createAVEC(v.val().val());
  VEC h;
  minus(v).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
}
TEST(AgradMixMatrixMinus, ffv_vector_1stDeriv) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::minus;

  vector_d d(3);
  vector_ffv v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  
  vector_d output_d;
  output_d = minus(d);
  EXPECT_FLOAT_EQ(100, output_d[0]);
  EXPECT_FLOAT_EQ(0, output_d[1]);
  EXPECT_FLOAT_EQ(-1, output_d[2]);

  vector_ffv output;
  output = minus(v);
  EXPECT_FLOAT_EQ(100, output[0].val_.val().val());
  EXPECT_FLOAT_EQ(0, output[1].val_.val().val());
  EXPECT_FLOAT_EQ(-1, output[2].val_.val().val());
  EXPECT_FLOAT_EQ(-1, output[0].d_.val().val());
  EXPECT_FLOAT_EQ(-1, output[1].d_.val().val());
  EXPECT_FLOAT_EQ(-1, output[2].d_.val().val());


  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val());
  VEC h;
  output[0].val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, ffv_vector_2ndDeriv_1) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::minus;

  vector_d d(3);
  vector_ffv v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  
  vector_d output_d;
  vector_ffv output;
  output = minus(v);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val());
  VEC h;
  output[0].val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, ffv_vector_2ndDeriv_2) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::minus;

  vector_d d(3);
  vector_ffv v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  
  vector_d output_d;
  vector_ffv output;
  output = minus(v);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val());
  VEC h;
  output[0].d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, ffv_vector_3rdDeriv) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::minus;

  vector_d d(3);
  vector_ffv v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  
  vector_d output_d;
  vector_ffv output;
  output = minus(v);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val());
  VEC h;
  output[0].d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, ffv_rowvector_1stDeriv) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;
  using stan::math::minus;

  row_vector_d d(3);
  row_vector_ffv v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  
  row_vector_d output_d;
  output_d = minus(d);
  EXPECT_FLOAT_EQ(100, output_d[0]);
  EXPECT_FLOAT_EQ(0, output_d[1]);
  EXPECT_FLOAT_EQ(-1, output_d[2]);

  row_vector_ffv output;
  output = minus(v);
  EXPECT_FLOAT_EQ(100, output[0].val_.val().val());
  EXPECT_FLOAT_EQ(0, output[1].val_.val().val());
  EXPECT_FLOAT_EQ(-1, output[2].val_.val().val());
  EXPECT_FLOAT_EQ(-1, output[0].d_.val().val());
  EXPECT_FLOAT_EQ(-1, output[1].d_.val().val());
  EXPECT_FLOAT_EQ(-1, output[2].d_.val().val());

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val());
  VEC h;
  output[0].val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, ffv_rowvector_2ndDeriv_1) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;
  using stan::math::minus;

  row_vector_d d(3);
  row_vector_ffv v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  
  row_vector_ffv output;
  output = minus(v);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val());
  VEC h;
  output[0].val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, ffv_rowvector_2ndDeriv_2) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;
  using stan::math::minus;

  row_vector_d d(3);
  row_vector_ffv v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  
  row_vector_ffv output;
  output = minus(v);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val());
  VEC h;
  output[0].d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, ffv_rowvector_3rdDeriv) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;
  using stan::math::minus;

  row_vector_d d(3);
  row_vector_ffv v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  
  row_vector_ffv output;
  output = minus(v);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val());
  VEC h;
  output[0].d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, ffv_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::minus;

  matrix_d d(2, 3);
  matrix_ffv v(2, 3);

  d << -100, 0, 1, 20, -40, 2;
  v << -100, 0, 1, 20, -40, 2;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(0,2).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
   v(1,2).d_ = 1.0;

  matrix_d output_d = minus(d);
  EXPECT_FLOAT_EQ(100, output_d(0,0));
  EXPECT_FLOAT_EQ(  0, output_d(0,1));
  EXPECT_FLOAT_EQ( -1, output_d(0,2));
  EXPECT_FLOAT_EQ(-20, output_d(1,0));
  EXPECT_FLOAT_EQ( 40, output_d(1,1));
  EXPECT_FLOAT_EQ( -2, output_d(1,2));

  matrix_ffv output = minus(v);
  EXPECT_FLOAT_EQ(100, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(  0, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val_.val().val());
  EXPECT_FLOAT_EQ(-20, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ( 40, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ( -2, output(1,2).val_.val().val());
  EXPECT_FLOAT_EQ( -1, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ( -1, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ( -1, output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ( -1, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ( -1, output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ( -1, output(1,2).d_.val().val());

  AVEC q = createAVEC(v(0,0).val().val(),v(0,1).val().val(),v(0,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixMinus, ffv_matrix_2ndDeriv_1) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::minus;

  matrix_d d(2, 3);
  matrix_ffv v(2, 3);

  d << -100, 0, 1, 20, -40, 2;
  v << -100, 0, 1, 20, -40, 2;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(0,2).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
   v(1,2).d_ = 1.0;

  matrix_ffv output = minus(v);

  AVEC q = createAVEC(v(0,0).val().val(),v(0,1).val().val(),v(0,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}

TEST(AgradMixMatrixMinus, ffv_matrix_2ndDeriv_2) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::minus;

  matrix_d d(2, 3);
  matrix_ffv v(2, 3);

  d << -100, 0, 1, 20, -40, 2;
  v << -100, 0, 1, 20, -40, 2;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(0,2).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
   v(1,2).d_ = 1.0;

  matrix_ffv output = minus(v);

  AVEC q = createAVEC(v(0,0).val().val(),v(0,1).val().val(),v(0,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}

TEST(AgradMixMatrixMinus, ffv_matrix_3rdDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::minus;

  matrix_d d(2, 3);
  matrix_ffv v(2, 3);

  d << -100, 0, 1, 20, -40, 2;
  v << -100, 0, 1, 20, -40, 2;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(0,2).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
   v(1,2).d_ = 1.0;

  matrix_ffv output = minus(v);

  AVEC q = createAVEC(v(0,0).val().val(),v(0,1).val().val(),v(0,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
