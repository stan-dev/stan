#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/sum.hpp>
#include <stan/math/rev/mat/fun/sum.hpp>
#include <stan/math/prim/mat/fun/sum.hpp>
#include <stan/math/prim/mat/fun/sum.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/rev/core.hpp>

using stan::math::fvar;
using stan::math::var;
TEST(AgradMixMatrixSum, fv_vector_1stDeriv) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::math::vector_fv;

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
  
  fvar<var> output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ( 0.0, output.d_.val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());  
  EXPECT_FLOAT_EQ( 6.0, output.d_.val());

  AVEC q = createAVEC(v(0).val(),v(1).val(),v(2).val(),v(3).val());
  VEC h;
  output.val_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
  EXPECT_FLOAT_EQ(1,h[3]);

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_.val());
  EXPECT_FLOAT_EQ(0.0, sum(v).d_.val());
}
TEST(AgradMixMatrixSum, fv_vector_2ndDeriv) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::math::vector_fv;

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
  
  fvar<var> output;
  output = sum(v);

  AVEC q = createAVEC(v(0).val(),v(1).val(),v(2).val(),v(3).val());
  VEC h;
  output.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixSum, fv_rowvector_1stDeriv) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

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
  
  fvar<var> output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ( 0.0, output.d_.val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());  
  EXPECT_FLOAT_EQ( 6.0, output.d_.val());

  AVEC q = createAVEC(v(0).val(),v(1).val(),v(2).val(),v(3).val());
  VEC h;
  output.val_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
  EXPECT_FLOAT_EQ(1,h[3]);

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_.val());
  EXPECT_FLOAT_EQ(0.0, sum(v).d_.val());
}
TEST(AgradMixMatrixSum, fv_rowvector_2ndDeriv) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

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
  
  fvar<var> output;
  output = sum(v);

  AVEC q = createAVEC(v(0).val(),v(1).val(),v(2).val(),v(3).val());
  VEC h;
  output.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixSum, fv_matrix_1stDeriv) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;

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
  
  fvar<var> output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ( 0.0, output.d_.val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ( 6.0, output.d_.val());

  AVEC q = createAVEC(v(0,0).val(),v(0,1).val(),v(1,0).val(),v(1,1).val());
  VEC h;
  output.val_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
  EXPECT_FLOAT_EQ(1,h[3]);

  d.resize(0, 0);
  v.resize(0, 0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_.val());
  EXPECT_FLOAT_EQ(0.0, sum(v).d_.val());
}
TEST(AgradMixMatrixSum, fv_matrix_2ndDeriv) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;

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
  
  fvar<var> output;
  output = sum(v);

  AVEC q = createAVEC(v(0,0).val(),v(0,1).val(),v(1,0).val(),v(1,1).val());
  VEC h;
  output.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixSum, ffv_vector_1stDeriv) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d d(6);
  vector_ffv v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
   v(5).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_.val().val());
  EXPECT_FLOAT_EQ( 0.0, output.d_.val().val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val().val());  
  EXPECT_FLOAT_EQ( 6.0, output.d_.val().val());

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  output.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
  EXPECT_FLOAT_EQ(1,h[3]);

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_.val().val());
  EXPECT_FLOAT_EQ(0.0, sum(v).d_.val().val());
}
TEST(AgradMixMatrixSum, ffv_vector_2ndDeriv_1) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d d(6);
  vector_ffv v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
   v(5).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = sum(v);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  output.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixSum, ffv_vector_2ndDeriv_2) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d d(6);
  vector_ffv v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
   v(5).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = sum(v);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  output.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixSum, ffv_vector_3rdDeriv) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d d(6);
  vector_ffv v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
   v(5).d_ = 1.0;
   v(0).val_.d_ = 1.0;
   v(1).val_.d_ = 1.0;
   v(2).val_.d_ = 1.0;
   v(3).val_.d_ = 1.0;
   v(4).val_.d_ = 1.0;
   v(5).val_.d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = sum(v);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  output.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixSum, ffv_rowvector_1stDeriv) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d(6);
  row_vector_ffv v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
   v(5).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_.val().val());
  EXPECT_FLOAT_EQ( 0.0, output.d_.val().val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val().val());  
  EXPECT_FLOAT_EQ( 6.0, output.d_.val().val());

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  output.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
  EXPECT_FLOAT_EQ(1,h[3]);

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_.val().val());
  EXPECT_FLOAT_EQ(0.0, sum(v).d_.val().val());
}
TEST(AgradMixMatrixSum, ffv_rowvector_2ndDeriv_1) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d(6);
  row_vector_ffv v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
   v(5).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = sum(v);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  output.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixSum, ffv_rowvector_2ndDeriv_2) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d(6);
  row_vector_ffv v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
   v(5).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = sum(v);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  output.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixSum, ffv_rowvector_3rdDeriv) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d(6);
  row_vector_ffv v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
   v(5).d_ = 1.0;
   v(0).val_.d_ = 1.0;
   v(1).val_.d_ = 1.0;
   v(2).val_.d_ = 1.0;
   v(3).val_.d_ = 1.0;
   v(4).val_.d_ = 1.0;
   v(5).val_.d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = sum(v);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  output.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixSum, ffv_matrix_1stDeriv) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d d(2, 3);
  matrix_ffv v(2, 3);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(0,2).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
   v(1,2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_.val().val());
  EXPECT_FLOAT_EQ( 0.0, output.d_.val().val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val().val());
  EXPECT_FLOAT_EQ( 6.0, output.d_.val().val());

  AVEC q = createAVEC(v(0,0).val().val(),v(0,1).val().val(),v(1,0).val().val(),v(1,1).val().val());
  VEC h;
  output.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
  EXPECT_FLOAT_EQ(1,h[3]);

  d.resize(0, 0);
  v.resize(0, 0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_.val().val());
  EXPECT_FLOAT_EQ(0.0, sum(v).d_.val().val());
}
TEST(AgradMixMatrixSum, ffv_matrix_2ndDeriv_1) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d d(2, 3);
  matrix_ffv v(2, 3);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(0,2).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
   v(1,2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = sum(v);

  AVEC q = createAVEC(v(0,0).val().val(),v(0,1).val().val(),v(1,0).val().val(),v(1,1).val().val());
  VEC h;
  output.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}

TEST(AgradMixMatrixSum, ffv_matrix_2ndDeriv_2) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d d(2, 3);
  matrix_ffv v(2, 3);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(0,2).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
   v(1,2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = sum(v);

  AVEC q = createAVEC(v(0,0).val().val(),v(0,1).val().val(),v(1,0).val().val(),v(1,1).val().val());
  VEC h;
  output.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}

TEST(AgradMixMatrixSum, ffv_matrix_3rdDeriv) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d d(2, 3);
  matrix_ffv v(2, 3);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(0,2).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
   v(1,2).d_ = 1.0;
   v(0,0).val_.d_ = 1.0;
   v(0,1).val_.d_ = 1.0;
   v(0,2).val_.d_ = 1.0;
   v(1,0).val_.d_ = 1.0;
   v(1,1).val_.d_ = 1.0;
   v(1,2).val_.d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = sum(v);

  AVEC q = createAVEC(v(0,0).val().val(),v(0,1).val().val(),v(1,0).val().val(),v(1,1).val().val());
  VEC h;
  output.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
