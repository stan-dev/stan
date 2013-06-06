#include <stan/agrad/fwd/matrix/crossprod.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFwdMatrix, crossprod_3x3_matrix) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
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
  matrix_d X = Z.transpose() * Z;
  matrix_fv output = stan::agrad::crossprod(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_);
  }

  EXPECT_FLOAT_EQ(28,output(0,0).d_);
  EXPECT_FLOAT_EQ(30,output(0,1).d_);
  EXPECT_FLOAT_EQ(26,output(0,2).d_);
  EXPECT_FLOAT_EQ(30,output(1,0).d_);
  EXPECT_FLOAT_EQ(32,output(1,1).d_);
  EXPECT_FLOAT_EQ(28,output(1,2).d_);
  EXPECT_FLOAT_EQ(26,output(2,0).d_);
  EXPECT_FLOAT_EQ(28,output(2,1).d_);
  EXPECT_FLOAT_EQ(24,output(2,2).d_);
}
TEST(AgradFwdMatrix, crossprod_2x2_matrix) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  matrix_d Z(2,2);
  Z <<3, 0,
     4, -3;
  matrix_fv Y(2,2);
  Y << 3, 0,
     4, -3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
  matrix_d X = Z.transpose() * Z;
  matrix_fv output = stan::agrad::crossprod(Y);
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_);
  }

  EXPECT_FLOAT_EQ( 28,output(0,0).d_);
  EXPECT_FLOAT_EQ(  8,output(0,1).d_);
  EXPECT_FLOAT_EQ(  8,output(1,0).d_);
  EXPECT_FLOAT_EQ(-12,output(1,1).d_);
}
TEST(AgradFwdMatrix, crossprod_1x1_matrix) {
  using stan::agrad::matrix_fv;

  matrix_fv Y(1,1);
  Y << 3;
   Y(0,0).d_ = 2.0;
  matrix_fv output = stan::agrad::crossprod(Y);
  EXPECT_FLOAT_EQ( 9, output(0,0).val_);
  EXPECT_FLOAT_EQ(12, output(0,0).d_);
}
TEST(AgradFwdMatrix, crossprod_1x3_matrix) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;

  matrix_fv Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
  matrix_fv output = stan::agrad::crossprod(Y);

  EXPECT_FLOAT_EQ(1, output(0,0).val_); 
  EXPECT_FLOAT_EQ(4,output(0,0).d_);
}
TEST(AgradFwdMatrix, crossprod_2x3_matrix) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;

  matrix_fv Y(2,3);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
  matrix_fv output = stan::agrad::crossprod(Y);

  EXPECT_FLOAT_EQ( 2, output(0,0).val_); 
  EXPECT_FLOAT_EQ(-2, output(0,1).val_); 
  EXPECT_FLOAT_EQ(-2, output(1,0).val_); 
  EXPECT_FLOAT_EQ(20, output(1,1).val_);
  EXPECT_FLOAT_EQ( 0, output(0,0).d_); 
  EXPECT_FLOAT_EQ(12, output(0,1).d_); 
  EXPECT_FLOAT_EQ(12, output(1,0).d_); 
  EXPECT_FLOAT_EQ(24, output(1,1).d_); 
}
TEST(AgradFwdMatrix, crossprod_3x2_matrix) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;

  matrix_fv Y(3,2);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_fv output = stan::agrad::crossprod(Y);

  EXPECT_FLOAT_EQ( 26, output(0,0).val_); 
  EXPECT_FLOAT_EQ(-37, output(0,1).val_); 
  EXPECT_FLOAT_EQ(-37, output(1,0).val_); 
  EXPECT_FLOAT_EQ( 86, output(1,1).val_);
  EXPECT_FLOAT_EQ( 32, output(0,0).d_); 
  EXPECT_FLOAT_EQ(  0, output(0,1).d_); 
  EXPECT_FLOAT_EQ(  0, output(1,0).d_); 
  EXPECT_FLOAT_EQ(-32, output(1,1).d_); 
}
TEST(AgradFwdFvarVarMatrix, crossprod_3x3_matrix) {
  using stan::agrad::matrix_fvv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);
  fvar<var> e(5.0,2.0);
  fvar<var> f(6.0,2.0);
  fvar<var> g(0.0,2.0);
  matrix_fvv Y(3,3);
  Y << a,g,g,b,c,g,d,e,f;

  matrix_d X = Z.transpose() * Z;
  matrix_fvv output = stan::agrad::crossprod(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ(28,output(0,0).d_.val());
  EXPECT_FLOAT_EQ(30,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(26,output(0,2).d_.val());
  EXPECT_FLOAT_EQ(30,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(32,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(28,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(26,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(28,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(24,output(2,2).d_.val());
}
TEST(AgradFwdFvarVarMatrix, crossprod_2x2_matrix) {
  using stan::agrad::matrix_fvv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d Z(2,2);
  Z <<3, 0,
     4, -3;

  fvar<var> a(3.0,2.0);
  fvar<var> b(0.0,2.0);
  fvar<var> c(4.0,2.0);
  fvar<var> d(-3.0,2.0);

  matrix_fvv Y(2,2);
  Y << a,b,c,d;
  matrix_d X = Z.transpose() * Z;
  matrix_fvv output = stan::agrad::crossprod(Y);
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 28,output(0,0).d_.val());
  EXPECT_FLOAT_EQ(  8,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(  8,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(-12,output(1,1).d_.val());
}
TEST(AgradFwdFvarVarMatrix, crossprod_1x1_matrix) {
  using stan::agrad::matrix_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(3.0,2.0);

  matrix_fvv Y(1,1);
  Y << a;
  matrix_fvv output = stan::agrad::crossprod(Y);
  EXPECT_FLOAT_EQ( 9, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(12, output(0,0).d_.val());
}
TEST(AgradFwdFvarVarMatrix, crossprod_1x3_matrix) {
  using stan::agrad::matrix_fvv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);

  matrix_fvv Y(1,3);
  Y << a,b,c;
  matrix_fvv output = stan::agrad::crossprod(Y);

  EXPECT_FLOAT_EQ(1, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(4,output(0,0).d_.val());
}
TEST(AgradFwdFvarVarMatrix, crossprod_2x3_matrix) {
  using stan::agrad::matrix_fvv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(-1.0,2.0);
  fvar<var> e(4.0,2.0);
  fvar<var> f(-9.0,2.0);

  matrix_fvv Y(2,3);
  Y << a,b,c,d,e,f;
  matrix_fvv output = stan::agrad::crossprod(Y);

  EXPECT_FLOAT_EQ( 2, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(-2, output(0,1).val_.val()); 
  EXPECT_FLOAT_EQ(-2, output(1,0).val_.val()); 
  EXPECT_FLOAT_EQ(20, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( 0, output(0,0).d_.val()); 
  EXPECT_FLOAT_EQ(12, output(0,1).d_.val()); 
  EXPECT_FLOAT_EQ(12, output(1,0).d_.val()); 
  EXPECT_FLOAT_EQ(24, output(1,1).d_.val()); 
}
TEST(AgradFwdFvarVarMatrix, crossprod_3x2_matrix) {
  using stan::agrad::matrix_fvv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(-1.0,2.0);
  fvar<var> e(4.0,2.0);
  fvar<var> f(-9.0,2.0);

  matrix_fvv Y(3,2);
  Y << a,b,c,d,e,f;
  matrix_fvv output = stan::agrad::crossprod(Y);

  EXPECT_FLOAT_EQ( 26, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(-37, output(0,1).val_.val()); 
  EXPECT_FLOAT_EQ(-37, output(1,0).val_.val()); 
  EXPECT_FLOAT_EQ( 86, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( 32, output(0,0).d_.val()); 
  EXPECT_FLOAT_EQ(  0, output(0,1).d_.val()); 
  EXPECT_FLOAT_EQ(  0, output(1,0).d_.val()); 
  EXPECT_FLOAT_EQ(-32, output(1,1).d_.val()); 
}
TEST(AgradFwdFvarFvarMatrix, crossprod_3x3_matrix) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;

  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
  fvar<fvar<double> > g;
  a.val_.val_ = 1.0;
  a.d_.val_ = 2.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 2.0;  
  e.val_.val_ = 5.0;
  e.d_.val_ = 2.0;
  f.val_.val_ = 6.0;
  f.d_.val_ = 2.0;
  g.val_.val_ = 0.0;
  g.d_.val_ = 2.0;

  matrix_ffv Y(3,3);
  Y << a,g,g,b,c,g,d,e,f;

  matrix_d X = Z.transpose() * Z;
  matrix_ffv output = stan::agrad::crossprod(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ(28,output(0,0).d_.val());
  EXPECT_FLOAT_EQ(30,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(26,output(0,2).d_.val());
  EXPECT_FLOAT_EQ(30,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(32,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(28,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(26,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(28,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(24,output(2,2).d_.val());
}
TEST(AgradFwdFvarFvarMatrix, crossprod_2x2_matrix) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;

  matrix_d Z(2,2);
  Z <<3, 0,
     4, -3;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  a.val_.val_ = 3.0;
  a.d_.val_ = 2.0;  
  b.val_.val_ = 0.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 4.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = -3.0;
  d.d_.val_ = 2.0;

  matrix_ffv Y(2,2);
  Y << a,b,c,d;
  matrix_d X = Z.transpose() * Z;
  matrix_ffv output = stan::agrad::crossprod(Y);
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 28,output(0,0).d_.val());
  EXPECT_FLOAT_EQ(  8,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(  8,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(-12,output(1,1).d_.val());
}
TEST(AgradFwdFvarFvarMatrix, crossprod_1x1_matrix) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;

  fvar<fvar<double> > a;
  a.val_.val_ = 3.0;
  a.d_.val_ = 2.0;

  matrix_ffv Y(1,1);
  Y << a;
  matrix_ffv output = stan::agrad::crossprod(Y);
  EXPECT_FLOAT_EQ( 9, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(12, output(0,0).d_.val());
}
TEST(AgradFwdFvarFvarMatrix, crossprod_1x3_matrix) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  a.val_.val_ = 1.0;
  a.d_.val_ = 2.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 2.0;

  matrix_ffv Y(1,3);
  Y << a,b,c;
  matrix_ffv output = stan::agrad::crossprod(Y);

  EXPECT_FLOAT_EQ(1, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(4,output(0,0).d_.val());
}
TEST(AgradFwdFvarFvarMatrix, crossprod_2x3_matrix) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 2.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = -1.0;
  d.d_.val_ = 2.0;  
  e.val_.val_ = 4.0;
  e.d_.val_ = 2.0;
  f.val_.val_ = -9.0;
  f.d_.val_ = 2.0;

  matrix_ffv Y(2,3);
  Y << a,b,c,d,e,f;
  matrix_ffv output = stan::agrad::crossprod(Y);

  EXPECT_FLOAT_EQ( 2, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(-2, output(0,1).val_.val()); 
  EXPECT_FLOAT_EQ(-2, output(1,0).val_.val()); 
  EXPECT_FLOAT_EQ(20, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( 0, output(0,0).d_.val()); 
  EXPECT_FLOAT_EQ(12, output(0,1).d_.val()); 
  EXPECT_FLOAT_EQ(12, output(1,0).d_.val()); 
  EXPECT_FLOAT_EQ(24, output(1,1).d_.val()); 
}
TEST(AgradFwdFvarFvarMatrix, crossprod_3x2_matrix) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 2.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = -1.0;
  d.d_.val_ = 2.0;  
  e.val_.val_ = 4.0;
  e.d_.val_ = 2.0;
  f.val_.val_ = -9.0;
  f.d_.val_ = 2.0;

  matrix_ffv Y(3,2);
  Y << a,b,c,d,e,f;
  matrix_ffv output = stan::agrad::crossprod(Y);

  EXPECT_FLOAT_EQ( 26, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(-37, output(0,1).val_.val()); 
  EXPECT_FLOAT_EQ(-37, output(1,0).val_.val()); 
  EXPECT_FLOAT_EQ( 86, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( 32, output(0,0).d_.val()); 
  EXPECT_FLOAT_EQ(  0, output(0,1).d_.val()); 
  EXPECT_FLOAT_EQ(  0, output(1,0).d_.val()); 
  EXPECT_FLOAT_EQ(-32, output(1,1).d_.val()); 
}
