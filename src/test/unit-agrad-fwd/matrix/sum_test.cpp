#include <stan/agrad/fwd/matrix/sum.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/sum.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

using stan::agrad::fvar;
using stan::agrad::var;
TEST(AgradFwdMatrixSum, fd_vector) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::agrad::vector_fd;

  vector_d d(6);
  vector_fd v(6);
  
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
TEST(AgradFwdMatrixSum, fd_rowvector) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;

  row_vector_d d(6);
  row_vector_fd v(6);
  
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
TEST(AgradFwdMatrixSum, fd_matrix) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;

  matrix_d d(2, 3);
  matrix_fd v(2, 3);
  
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
TEST(AgradFwdMatrixSum, fv_vector_1stDeriv) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

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
TEST(AgradFwdMatrixSum, fv_vector_2ndDeriv) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

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
TEST(AgradFwdMatrixSum, fv_rowvector_1stDeriv) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

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
TEST(AgradFwdMatrixSum, fv_rowvector_2ndDeriv) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

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
TEST(AgradFwdMatrixSum, fv_matrix_1stDeriv) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

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
TEST(AgradFwdMatrixSum, fv_matrix_2ndDeriv) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

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
TEST(AgradFwdMatrixSum, ffd_vector) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;

  vector_d d(6);
  vector_ffd v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
   v(5).d_ = 1.0;
  
  fvar<fvar<double> > output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ( 0.0, output.d_.val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());  
  EXPECT_FLOAT_EQ( 6.0, output.d_.val());

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_.val());
  EXPECT_FLOAT_EQ(0.0, sum(v).d_.val());
}
TEST(AgradFwdMatrixSum, ffd_rowvector) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;

  row_vector_d d(6);
  row_vector_ffd v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
   v(5).d_ = 1.0;
  
  fvar<fvar<double> > output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ( 0.0, output.d_.val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());  
  EXPECT_FLOAT_EQ( 6.0, output.d_.val());

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_.val());
  EXPECT_FLOAT_EQ(0.0, sum(v).d_.val());
}
TEST(AgradFwdMatrixSum, ffd_matrix) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;

  matrix_d d(2, 3);
  matrix_ffd v(2, 3);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(0,2).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
   v(1,2).d_ = 1.0;
  
  fvar<fvar<double> > output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ( 0.0, output.d_.val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ( 6.0, output.d_.val());

  d.resize(0, 0);
  v.resize(0, 0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_.val());
  EXPECT_FLOAT_EQ(0.0, sum(v).d_.val());
}
TEST(AgradFwdMatrixSum, ffv_vector_1stDeriv) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

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
TEST(AgradFwdMatrixSum, ffv_vector_2ndDeriv_1) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

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
TEST(AgradFwdMatrixSum, ffv_vector_2ndDeriv_2) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

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
TEST(AgradFwdMatrixSum, ffv_vector_3rdDeriv) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

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
TEST(AgradFwdMatrixSum, ffv_rowvector_1stDeriv) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;

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
TEST(AgradFwdMatrixSum, ffv_rowvector_2ndDeriv_1) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;

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
TEST(AgradFwdMatrixSum, ffv_rowvector_2ndDeriv_2) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;

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
TEST(AgradFwdMatrixSum, ffv_rowvector_3rdDeriv) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;

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
TEST(AgradFwdMatrixSum, ffv_matrix_1stDeriv) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;

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
TEST(AgradFwdMatrixSum, ffv_matrix_2ndDeriv_1) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;

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

TEST(AgradFwdMatrixSum, ffv_matrix_2ndDeriv_2) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;

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

TEST(AgradFwdMatrixSum, ffv_matrix_3rdDeriv) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;

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
