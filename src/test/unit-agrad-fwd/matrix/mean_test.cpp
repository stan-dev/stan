#include <stan/math/matrix/mean.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixMean, fd_vector) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_fd;
  using stan::agrad::fvar;

  vector_d d1(3);
  vector_fd v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<double> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(0.0, output.d_);
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(1.0, output.d_);
}
TEST(AgradFwdMatrixMean, fd_vector_exception) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_fd;

  vector_d d;
  vector_fd v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrixMean, fd_rowvector) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;
  using stan::agrad::fvar;

  row_vector_d d1(3);
  row_vector_fd v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<double> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(0.0, output.d_);
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(1.0, output.d_);
}
TEST(AgradFwdMatrixMean, fd_rowvector_exception) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;

  row_vector_d d;
  row_vector_fd v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrixMean, fd_matrix) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;
  using stan::agrad::fvar;

  matrix_d d1(3,1);
  matrix_fd v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<double> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(0.0, output.d_);
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(1.0, output.d_);
}
TEST(AgradFwdMatrixMean, fd_matrix_exception) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;
 
  matrix_d d;
  matrix_fd v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrixMean, fd_StdVector) {
  using stan::math::mean;
  using stan::agrad::fvar;

  std::vector<fvar<double> > x(0);
  EXPECT_THROW(mean(x), std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0, mean(x).val_);
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(1.5, mean(x).val_);

  std::vector<fvar<double> > y;
  fvar<double> a = 1.0;
  a.d_ = 1.0;
  fvar<double> b = 2.0;
  b.d_ = 1.0;
  y.push_back(a);
  y.push_back(b);
  fvar<double> f = mean(y);

  EXPECT_FLOAT_EQ(1.5, f.val_);
  EXPECT_FLOAT_EQ(1.0, f.d_);
}
TEST(AgradFwdMatrixMean, fv_vector_1stDeriv) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d d1(3);
  vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<var> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output.val_.grad(q,h);
  EXPECT_FLOAT_EQ(1.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(1.0/3.0,h[1]);
  EXPECT_FLOAT_EQ(1.0/3.0,h[2]);
}
TEST(AgradFwdMatrixMean, fv_vector_2ndDeriv) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d d1(3);
  vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<var> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMean, fv_vector_exception) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d;
  vector_fv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrixMean, fv_rowvector_1stDeriv) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<var> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output.val_.grad(q,h);
  EXPECT_FLOAT_EQ(1.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(1.0/3.0,h[1]);
  EXPECT_FLOAT_EQ(1.0/3.0,h[2]);
}
TEST(AgradFwdMatrixMean, fv_rowvector_2ndDeriv) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<var> output;
  output = mean(v1);

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMean, fv_rowvector_exception) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d;
  row_vector_fv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrixMean, fv_matrix_1stDeriv) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_fv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<var> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());

  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(0,2).val());
  VEC h;
  output.val_.grad(q,h);
  EXPECT_FLOAT_EQ(1.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(1.0/3.0,h[1]);
  EXPECT_FLOAT_EQ(1.0/3.0,h[2]);
}
TEST(AgradFwdMatrixMean, fv_matrix_2ndDeriv) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_fv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<var> output;
  output = mean(v1);

  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(0,2).val());
  VEC h;
  output.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMean, fv_matrix_exception) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
 
  matrix_d d;
  matrix_fv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrixMean, fv_StdVector_1stDeriv) {
  using stan::math::mean;
  using stan::agrad::fvar;
  using stan::agrad::var;

  std::vector<fvar<var> > x(0);
  EXPECT_THROW(mean(x), std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0, mean(x).val_.val());
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(1.5, mean(x).val_.val());

  std::vector<fvar<var> > y;
  fvar<var> a = 1.0;
  a.d_ = 1.0;
  fvar<var> b = 2.0;
  b.d_ = 1.0;
  y.push_back(a);
  y.push_back(b);
  fvar<var> f = mean(y);

  EXPECT_FLOAT_EQ(1.5, f.val_.val());
  EXPECT_FLOAT_EQ(1.0, f.d_.val());

  AVEC q = createAVEC(a.val(),b.val());
  VEC h;
  f.val_.grad(q,h);
  EXPECT_FLOAT_EQ(1.0/2.0,h[0]);
  EXPECT_FLOAT_EQ(1.0/2.0,h[1]);
}
TEST(AgradFwdMatrixMean, fv_StdVector_2ndDeriv) {
  using stan::math::mean;
  using stan::agrad::fvar;
  using stan::agrad::var;

  std::vector<fvar<var> > y;
  fvar<var> a = 1.0;
  a.d_ = 1.0;
  fvar<var> b = 2.0;
  b.d_ = 1.0;
  y.push_back(a);
  y.push_back(b);
  fvar<var> f = mean(y);

  AVEC q = createAVEC(a.val(),b.val());
  VEC h;
  f.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
}
TEST(AgradFwdMatrixMean, ffd_vector) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;

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
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());
}
TEST(AgradFwdMatrixMean, ffd_vector_exception) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;

  vector_d d;
  vector_ffd v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrixMean, ffd_rowvector) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;

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
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());
}
TEST(AgradFwdMatrixMean, ffd_rowvector_exception) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;

  row_vector_d d;
  row_vector_ffd v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrixMean, ffd_matrix) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
  using stan::agrad::fvar;

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
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());
}
TEST(AgradFwdMatrixMean, ffd_matrix_exception) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
 
  matrix_d d;
  matrix_ffd v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrixMean, ffd_StdVector) {
  using stan::math::mean;
  using stan::agrad::fvar;

  std::vector<fvar<fvar<double> > > x(0);
  EXPECT_THROW(mean(x), std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0, mean(x).val_.val());
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(1.5, mean(x).val_.val());

  std::vector<fvar<fvar<double> > > y;
  fvar<fvar<double> > a,b;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  y.push_back(a);
  y.push_back(b);
  fvar<fvar<double> > f = mean(y);

  EXPECT_FLOAT_EQ(1.5, f.val_.val());
  EXPECT_FLOAT_EQ(1.0, f.d_.val());
}
TEST(AgradFwdMatrixMean, ffv_vector_1stDeriv) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d d1(3);
  vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val().val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val().val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val().val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val().val());

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(1.0/3.0,h[1]);
  EXPECT_FLOAT_EQ(1.0/3.0,h[2]);
}
TEST(AgradFwdMatrixMean, ffv_vector_2ndDeriv_1) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d d1(3);
  vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = mean(v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMean, ffv_vector_2ndDeriv_2) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d d1(3);
  vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = mean(v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMean, ffv_vector_3rdDeriv) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d d1(3);
  vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_.val_ = 1.0;
   v1(1).d_.val_ = 1.0;
   v1(2).d_.val_ = 1.0;
   v1(0).val_.d_ = 1.0;
   v1(1).val_.d_ = 1.0;
   v1(2).val_.d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = mean(v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMean, ffv_vector_exception) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

  vector_d d;
  vector_ffv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrixMean, ffv_rowvector_1stDeriv) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val().val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val().val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val().val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val().val());

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(1.0/3.0,h[1]);
  EXPECT_FLOAT_EQ(1.0/3.0,h[2]);
}
TEST(AgradFwdMatrixMean, ffv_rowvector_2ndDeriv_1) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = mean(v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMean, ffv_rowvector_2ndDeriv_2) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = mean(v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMean, ffv_rowvector_3rdDeriv) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_.val_ = 1.0;
   v1(1).d_.val_ = 1.0;
   v1(2).d_.val_ = 1.0;
   v1(0).val_.d_ = 1.0;
   v1(1).val_.d_ = 1.0;
   v1(2).val_.d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = mean(v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMean, ffv_rowvector_exception) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;

  row_vector_d d;
  row_vector_ffv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrixMean, ffv_matrix_1stDeriv) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_ffv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val().val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val().val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val().val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val().val());

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(0,2).val().val());
  VEC h;
  output.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.0/3.0,h[0]);
  EXPECT_FLOAT_EQ(1.0/3.0,h[1]);
  EXPECT_FLOAT_EQ(1.0/3.0,h[2]);
}
TEST(AgradFwdMatrixMean, ffv_matrix_2ndDeriv_1) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_ffv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = mean(v1);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(0,2).val().val());
  VEC h;
  output.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMean, ffv_matrix_2ndDeriv_2) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_ffv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = mean(v1);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(0,2).val().val());
  VEC h;
  output.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMean, ffv_matrix_3rdDeriv) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_ffv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_.val_ = 1.0;
   v1(1).d_.val_ = 1.0;
   v1(2).d_.val_ = 1.0;
   v1(0).val_.d_ = 1.0;
   v1(1).val_.d_ = 1.0;
   v1(2).val_.d_ = 1.0;
  
  fvar<fvar<var> > output;
  output = mean(v1);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(0,2).val().val());
  VEC h;
  output.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixMean, ffv_matrix_exception) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
 
  matrix_d d;
  matrix_ffv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrixMean, ffv_StdVector_1stDeriv) {
  using stan::math::mean;
  using stan::agrad::fvar;
  using stan::agrad::var;

  std::vector<fvar<fvar<var> > > x(0);
  EXPECT_THROW(mean(x), std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0, mean(x).val_.val().val());
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(1.5, mean(x).val_.val().val());

  std::vector<fvar<fvar<var> > > y;
  fvar<fvar<var> > a = 1.0;
  a.d_ = 1.0;
  fvar<fvar<var> > b = 2.0;
  b.d_ = 1.0;
  y.push_back(a);
  y.push_back(b);
  fvar<fvar<var> > f = mean(y);

  EXPECT_FLOAT_EQ(1.5, f.val_.val().val());
  EXPECT_FLOAT_EQ(1.0, f.d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val());
  VEC h;
  f.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.0/2.0,h[0]);
  EXPECT_FLOAT_EQ(1.0/2.0,h[1]);
}
TEST(AgradFwdMatrixMean, ffv_StdVector_2ndDeriv_1) {
  using stan::math::mean;
  using stan::agrad::fvar;
  using stan::agrad::var;

  std::vector<fvar<fvar<var> > > y;
  fvar<fvar<var> > a = 1.0;
  a.d_ = 1.0;
  fvar<fvar<var> > b = 2.0;
  b.d_ = 1.0;
  y.push_back(a);
  y.push_back(b);
  fvar<fvar<var> > f = mean(y);

  AVEC q = createAVEC(a.val().val(),b.val().val());
  VEC h;
  f.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
}

TEST(AgradFwdMatrixMean, ffv_StdVector_2ndDeriv_2) {
  using stan::math::mean;
  using stan::agrad::fvar;
  using stan::agrad::var;

  std::vector<fvar<fvar<var> > > y;
  fvar<fvar<var> > a = 1.0;
  a.d_ = 1.0;
  fvar<fvar<var> > b = 2.0;
  b.d_ = 1.0;
  y.push_back(a);
  y.push_back(b);
  fvar<fvar<var> > f = mean(y);

  AVEC q = createAVEC(a.val().val(),b.val().val());
  VEC h;
  f.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
}
TEST(AgradFwdMatrixMean, ffv_StdVector_3rdDeriv) {
  using stan::math::mean;
  using stan::agrad::fvar;
  using stan::agrad::var;

  std::vector<fvar<fvar<var> > > y;
  fvar<fvar<var> > a = 1.0;
  a.d_.val_ = 1.0;
  a.val_.d_ = 1.0;
  fvar<fvar<var> > b = 2.0;
  b.d_.val_ = 1.0;
  b.val_.d_ = 1.0;

  y.push_back(a);
  y.push_back(b);
  fvar<fvar<var> > f = mean(y);

  AVEC q = createAVEC(a.val().val(),b.val().val());
  VEC h;
  f.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
}
