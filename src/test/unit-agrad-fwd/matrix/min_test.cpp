#include <stan/math/matrix/min.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

using stan::agrad::fvar;
TEST(AgradFwdMatrixMin, fd_vector) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::agrad::vector_fd;
  
  vector_d d1(3);
  vector_fd v1(3);
  
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
TEST(AgradFwdMatrixMin, fd_vector_exception) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::agrad::vector_fd;

  vector_d d;
  vector_fd v;
  d.resize(0);
  v.resize(0);
  EXPECT_EQ(std::numeric_limits<double>::infinity(), min(v).val_);
  EXPECT_EQ(0, min(v).d_);
}
TEST(AgradFwdMatrixMin, fd_rowvector) {
  using stan::math::min;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;

  row_vector_d d1(3);
  row_vector_fd v1(3);
  
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
TEST(AgradFwdMatrixMin, fd_rowvector_exception) {
  using stan::math::min;
  using stan::agrad::row_vector_fd;

  row_vector_fd v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_);
  EXPECT_FLOAT_EQ(0, min(v).d_);
}
TEST(AgradFwdMatrixMin, fd_matrix) {
  using stan::math::min;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;
 
  matrix_d d1(3,1);
  matrix_fd v1(1,3);
  
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
TEST(AgradFwdMatrixMin, fd_matrix_exception) {
  using stan::math::min;
  using stan::agrad::matrix_fd;

  matrix_fd v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_);
  EXPECT_EQ(0, min(v).d_);
}
TEST(AgradFwdMatrixMin, fv_vector_1stDeriv) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::var;

  vector_d d1(3);
  vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<var> output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output.val_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
}
TEST(AgradFwdMatrixMin, fv_vector_2ndDeriv) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::var;

  vector_d d1(3);
  vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<var> output;
  output = min(v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMin, fv_vector_exception) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d;
  vector_fv v;
  d.resize(0);
  v.resize(0);
  EXPECT_EQ(std::numeric_limits<double>::infinity(), min(v).val_.val());
  EXPECT_EQ(0, min(v).d_.val());
}
TEST(AgradFwdMatrixMin, fv_rowvector_1stDeriv) {
  using stan::math::min;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<var> output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output.val_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
}
TEST(AgradFwdMatrixMin, fv_rowvector_2ndDeriv) {
  using stan::math::min;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<var> output;
  output = min(v1);

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMin, fv_rowvector_exception) {
  using stan::math::min;
  using stan::agrad::row_vector_fv;

  row_vector_fv v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_.val());
  EXPECT_FLOAT_EQ(0, min(v).d_.val());
}
TEST(AgradFwdMatrixMin, fv_matrix_1stDeriv) {
  using stan::math::min;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_fv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<var> output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());

  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(0,2).val());
  VEC h;
  output.val_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
}
TEST(AgradFwdMatrixMin, fv_matrix_2ndDeriv) {
  using stan::math::min;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_fv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<var> output;
  output = min(v1);

  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(0,2).val());
  VEC h;
  output.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMin, fv_matrix_exception) {
  using stan::math::min;
  using stan::agrad::matrix_fv;

  matrix_fv v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_.val());
  EXPECT_EQ(0, min(v).d_.val());
}
TEST(AgradFwdMatrixMin, ffd_vector) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;
  
  vector_d d1(3);
  vector_ffd v1(3);
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  b.val_.val_ = 0.0;
  c.val_.val_ = -3.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<fvar<double> > output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdMatrixMin, ffd_vector_exception) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;

  vector_d d;
  vector_ffd v;
  d.resize(0);
  v.resize(0);
  EXPECT_EQ(std::numeric_limits<double>::infinity(), min(v).val_.val());
  EXPECT_EQ(0, min(v).d_.val());
}
TEST(AgradFwdMatrixMin, ffd_rowvector) {
  using stan::math::min;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;

  row_vector_d d1(3);
  row_vector_ffd v1(3);
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  b.val_.val_ = 0.0;
  c.val_.val_ = -3.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<fvar<double> > output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdMatrixMin, ffd_rowvector_exception) {
  using stan::math::min;
  using stan::agrad::row_vector_ffd;

  row_vector_ffd v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_.val());
  EXPECT_FLOAT_EQ(0, min(v).d_.val());
}
TEST(AgradFwdMatrixMin, ffd_matrix) {
  using stan::math::min;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
 
  matrix_d d1(3,1);
  matrix_ffd v1(1,3);
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  b.val_.val_ = 0.0;
  c.val_.val_ = -3.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;

  fvar<fvar<double> > output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdMatrixMin, ffd_matrix_exception) {
  using stan::math::min;
  using stan::agrad::matrix_ffd;

  matrix_ffd v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_.val());
  EXPECT_EQ(0, min(v).d_.val());
}
TEST(AgradFwdMatrixMin, ffv_vector_1stDeriv) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::var;

  vector_d d1(3);
  vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_.val().val());
  EXPECT_FLOAT_EQ(0, output.d_.val().val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_.val().val());
  EXPECT_FLOAT_EQ(1, output.d_.val().val());

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
}
TEST(AgradFwdMatrixMin, ffv_vector_2ndDeriv_1) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::var;

  vector_d d1(3);
  vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = min(v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMin, ffv_vector_2ndDeriv_2) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::var;

  vector_d d1(3);
  vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = min(v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMin, ffv_vector_3rdDeriv) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::var;

  vector_d d1(3);
  vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
   v1(0).val_.d_ = 1.0;
   v1(1).val_.d_ = 1.0;
   v1(2).val_.d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = min(v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMin, ffv_vector_exception) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

  vector_d d;
  vector_ffv v;
  d.resize(0);
  v.resize(0);
  EXPECT_EQ(std::numeric_limits<double>::infinity(), min(v).val_.val().val());
  EXPECT_EQ(0, min(v).d_.val().val());
}
TEST(AgradFwdMatrixMin, ffv_rowvector_1stDeriv) {
  using stan::math::min;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_.val().val());
  EXPECT_FLOAT_EQ(0, output.d_.val().val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_.val().val());
  EXPECT_FLOAT_EQ(1, output.d_.val().val());

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
}
TEST(AgradFwdMatrixMin, ffv_rowvector_2ndDeriv_1) {
  using stan::math::min;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = min(v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMin, ffv_rowvector_2ndDeriv_2) {
  using stan::math::min;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = min(v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMin, ffv_rowvector_3rdDeriv) {
  using stan::math::min;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
   v1(0).val_.d_ = 1.0;
   v1(1).val_.d_ = 1.0;
   v1(2).val_.d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = min(v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMin, ffv_rowvector_exception) {
  using stan::math::min;
  using stan::agrad::row_vector_ffv;

  row_vector_ffv v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_.val().val());
  EXPECT_FLOAT_EQ(0, min(v).d_.val().val());
}
TEST(AgradFwdMatrixMin, ffv_matrix_1stDeriv) {
  using stan::math::min;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_ffv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_.val().val());
  EXPECT_FLOAT_EQ(0, output.d_.val().val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_.val().val());
  EXPECT_FLOAT_EQ(1, output.d_.val().val());

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(0,2).val().val());
  VEC h;
  output.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
}
TEST(AgradFwdMatrixMin, ffv_matrix_2ndDeriv_1) {
  using stan::math::min;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_ffv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = min(v1);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(0,2).val().val());
  VEC h;
  output.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMin, ffv_matrix_2ndDeriv_2) {
  using stan::math::min;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_ffv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = min(v1);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(0,2).val().val());
  VEC h;
  output.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMin, ffv_matrix_3rdDeriv) {
  using stan::math::min;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_ffv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
   v1(0,0).val_.d_ = 1.0;
   v1(0,1).val_.d_ = 1.0;
   v1(0,2).val_.d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = min(v1);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(0,2).val().val());
  VEC h;
  output.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMin, ffv_matrix_exception) {
  using stan::math::min;
  using stan::agrad::matrix_ffv;

  matrix_ffv v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_.val().val());
  EXPECT_EQ(0, min(v).d_.val().val());
}
