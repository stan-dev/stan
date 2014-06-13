#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/math/matrix/log_determinant.hpp>
#include <stan/agrad/fwd/matrix/log_determinant.hpp>
#include <stan/math/matrix/log_determinant.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixLogDeterminant,fd) {
  using stan::agrad::matrix_fd;
  using stan::agrad::fvar;
  using stan::agrad::log_determinant;
  
  matrix_fd v(2,2);
  v << 0, 1, 2, 3;
  v(0,0).d_ = 1.0;
  v(0,1).d_ = 2.0;
  v(1,0).d_ = 2.0;
  v(1,1).d_ = 2.0;
  
  fvar<double> det;
  det = log_determinant(v);
  EXPECT_FLOAT_EQ(std::log(2.0), det.val_);
  EXPECT_FLOAT_EQ(1.5, det.d_);
}

TEST(AgradFwdMatrixLogDeterminant,fd_exception) {
  using stan::agrad::matrix_fd;
  using stan::math::log_determinant;
  
  EXPECT_THROW(log_determinant(matrix_fd(2,3)), std::domain_error);
}
TEST(AgradFwdMatrixLogDeterminant,fv_1stDeriv) {
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_determinant;
  
  fvar<var> a(0.0,1.0);
  fvar<var> b(1.0,2.0);
  fvar<var> c(2.0,2.0);
  fvar<var> d(3.0,2.0);

  matrix_fv v(2,2);
  v << a,b,c,d;
  
  fvar<var> det;
  det = log_determinant(v);
  EXPECT_FLOAT_EQ(std::log(2.0), det.val_.val());
  EXPECT_FLOAT_EQ(1.5, det.d_.val());

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  det.val_.grad(q,h);
  EXPECT_FLOAT_EQ(-1.5,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
  EXPECT_FLOAT_EQ(.5,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixLogDeterminant,fv_2ndDeriv) {
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_determinant;
  
  fvar<var> a(0.0,1.0);
  fvar<var> b(1.0,2.0);
  fvar<var> c(2.0,2.0);
  fvar<var> d(3.0,2.0);
  matrix_fv v(2,2);
  v << a,b,c,d;
  
  fvar<var> det;
  det = log_determinant(v);

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  det.d_.grad(q,h);
  EXPECT_FLOAT_EQ(1.25,h[0]);
  EXPECT_FLOAT_EQ(-.5,h[1]);
  EXPECT_FLOAT_EQ(0.25,h[2]);
  EXPECT_FLOAT_EQ(-.5,h[3]);
}
TEST(AgradFwdMatrixLogDeterminant,fv_exception) {
  using stan::agrad::matrix_fv;
  using stan::math::log_determinant;
  
  EXPECT_THROW(log_determinant(matrix_fv(2,3)), std::domain_error);
}
TEST(AgradFwdMatrixLogDeterminant,ffd) {
  using stan::agrad::matrix_ffd;
  using stan::agrad::fvar;
  using stan::agrad::log_determinant;
  
  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 0.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 1.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 2.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = 3.0;
  d.d_.val_ = 2.0; 

  matrix_ffd v(2,2);
  v << a,b,c,d;
  
  fvar<fvar<double> > det;
  det = log_determinant(v);
  EXPECT_FLOAT_EQ(std::log(2.0), det.val_.val());
  EXPECT_FLOAT_EQ(1.5, det.d_.val());
}

TEST(AgradFwdMatrixLogDeterminant,ffd_exception) {
  using stan::agrad::matrix_ffd;
  using stan::math::log_determinant;
  
  EXPECT_THROW(log_determinant(matrix_ffd(2,3)), std::domain_error);
}
TEST(AgradFwdMatrixLogDeterminant,ffv_1stDeriv) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_determinant;
  
  fvar<fvar<var> > a(0.0,1.0);
  fvar<fvar<var> > b(1.0,2.0);
  fvar<fvar<var> > c(2.0,2.0);
  fvar<fvar<var> > d(3.0,2.0);

  matrix_ffv v(2,2);
  v << a,b,c,d;
  
  fvar<fvar<var> > det;
  det = log_determinant(v);
  EXPECT_FLOAT_EQ(std::log(2.0), det.val_.val().val());
  EXPECT_FLOAT_EQ(1.5, det.d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  det.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-1.5,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
  EXPECT_FLOAT_EQ(.5,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixLogDeterminant,ffv_2ndDeriv_1) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_determinant;
  
  fvar<fvar<var> > a(0.0,1.0);
  fvar<fvar<var> > b(1.0,2.0);
  fvar<fvar<var> > c(2.0,2.0);
  fvar<fvar<var> > d(3.0,2.0);
  matrix_ffv v(2,2);
  v << a,b,c,d;
  
  fvar<fvar<var> > det;
  det = log_determinant(v);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  det.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixLogDeterminant,ffv_2ndDeriv_2) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_determinant;
  
  fvar<fvar<var> > a(0.0,1.0);
  fvar<fvar<var> > b(1.0,2.0);
  fvar<fvar<var> > c(2.0,2.0);
  fvar<fvar<var> > d(3.0,2.0);
  matrix_ffv v(2,2);
  v << a,b,c,d;
  
  fvar<fvar<var> > det;
  det = log_determinant(v);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  det.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.25,h[0]);
  EXPECT_FLOAT_EQ(-.5,h[1]);
  EXPECT_FLOAT_EQ(0.25,h[2]);
  EXPECT_FLOAT_EQ(-.5,h[3]);
}
TEST(AgradFwdMatrixLogDeterminant,ffv_3rdDeriv) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_determinant;
  
  fvar<fvar<var> > a(0.0,1.0);
  fvar<fvar<var> > b(1.0,2.0);
  fvar<fvar<var> > c(2.0,2.0);
  fvar<fvar<var> > d(3.0,2.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;

  matrix_ffv v(2,2);
  v << a,b,c,d;
  
  fvar<fvar<var> > det;
  det = log_determinant(v);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  det.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(1.5,h[0]);
  EXPECT_FLOAT_EQ(-1.25,h[1]);
  EXPECT_FLOAT_EQ(-1,h[2]);
  EXPECT_FLOAT_EQ(0.75,h[3]);
}
TEST(AgradFwdMatrixLogDeterminant,ffv_exception) {
  using stan::agrad::matrix_ffv;
  using stan::math::log_determinant;
  
  EXPECT_THROW(log_determinant(matrix_ffv(2,3)), std::domain_error);
}
