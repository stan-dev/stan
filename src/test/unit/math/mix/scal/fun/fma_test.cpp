#include <cmath>
#include <gtest/gtest.h>
#include <stan/math/fwd/scal/fun/fma.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/rev/scal/fun/fma.hpp>



TEST(AgradFwdFma,FvarVar_FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  fvar<var> x(2.5,1.3);
  fvar<var> y(1.7,1.5);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(2.5 * 1.5 + 1.3 * 1.7 + 1.0, a.d_.val());

  AVEC w = createAVEC(x.val_,y.val_,z.val_);
  VEC g;
  a.val_.grad(w,g);
  EXPECT_FLOAT_EQ(1.7, g[0]);
  EXPECT_FLOAT_EQ(2.5,g[1]);
  EXPECT_FLOAT_EQ(1.0,g[2]);
}
TEST(AgradFwdFma,FvarVar_FvarVar_double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  fvar<var> x(2.5,1.3);
  fvar<var> y(1.7,1.5);
  double z(1.5);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(2.5 * 1.5 + 1.3 * 1.7, a.d_.val());

  AVEC w = createAVEC(x.val_,y.val_);
  VEC g;
  a.val_.grad(w,g);
  EXPECT_FLOAT_EQ(1.7, g[0]);
  EXPECT_FLOAT_EQ(2.5,g[1]);
}
TEST(AgradFwdFma,FvarVar_double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  fvar<var> x(2.5,1.3);
  double y(1.7);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * 1.7 + 1.0, a.d_.val());

  AVEC w = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(w,g);
  EXPECT_FLOAT_EQ(1.7, g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
}
TEST(AgradFwdFma,FvarVar_double_double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  fvar<var> x(2.5,1.3);
  double y(1.7);
  double z(1.5);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * 1.7, a.d_.val());

  AVEC w = createAVEC(x.val_);
  VEC g;
  a.val_.grad(w,g);
  EXPECT_FLOAT_EQ(1.7, g[0]);
}
TEST(AgradFwdFma,Double_FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  double x(2.5);
  fvar<var> y(1.7,1.5);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(2.5 * 1.5 + 1.0, a.d_.val());

  AVEC w = createAVEC(y.val_,z.val_);
  VEC g;
  a.val_.grad(w,g);
  EXPECT_FLOAT_EQ(2.5,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
}
TEST(AgradFwdFma,Double_Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  double x(2.5); 
  double y(1.7);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.0, a.d_.val());

  AVEC w = createAVEC(z.val_);
  VEC g;
  a.val_.grad(w,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}
TEST(AgradFwdFma,Double_FvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  double x(2.5);
  fvar<var> y(1.7,1.5);
  double z(1.5);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(2.5 * 1.5, a.d_.val());

  AVEC w = createAVEC(y.val_);
  VEC g;
  a.val_.grad(w,g);
  EXPECT_FLOAT_EQ(2.5,g[0]);
}

TEST(AgradFwdFma,FvarVar_FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  fvar<var> x(2.5,1.3);
  fvar<var> y(1.7,1.5);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(2.5 * 1.5 + 1.3 * 1.7 + 1.0, a.d_.val());

  AVEC w = createAVEC(x.val_,y.val_,z.val_);
  VEC g;
  a.d_.grad(w,g);
  EXPECT_FLOAT_EQ(1.5, g[0]);
  EXPECT_FLOAT_EQ(1.3,g[1]);
  EXPECT_FLOAT_EQ(0,g[2]);
}
TEST(AgradFwdFma,FvarVar_FvarVar_double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  fvar<var> x(2.5,1.3);
  fvar<var> y(1.7,1.5);
  double z(1.5);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(2.5 * 1.5 + 1.3 * 1.7, a.d_.val());

  AVEC w = createAVEC(x.val_,y.val_);
  VEC g;
  a.d_.grad(w,g);
  EXPECT_FLOAT_EQ(1.5, g[0]);
  EXPECT_FLOAT_EQ(1.3,g[1]);
}
TEST(AgradFwdFma,FvarVar_double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  fvar<var> x(2.5,1.3);
  double y(1.7);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * 1.7 + 1.0, a.d_.val());

  AVEC w = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(w,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(0,g[1]);
}
TEST(AgradFwdFma,FvarVar_double_double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  fvar<var> x(2.5,1.3);
  double y(1.7);
  double z(1.5);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * 1.7, a.d_.val());

  AVEC w = createAVEC(x.val_);
  VEC g;
  a.d_.grad(w,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdFma,Double_FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  double x(2.5);
  fvar<var> y(1.7,1.5);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(2.5 * 1.5 + 1.0, a.d_.val());

  AVEC w = createAVEC(y.val_,z.val_);
  VEC g;
  a.d_.grad(w,g);
  EXPECT_FLOAT_EQ(0,g[0]);
  EXPECT_FLOAT_EQ(0,g[1]);
}
TEST(AgradFwdFma,Double_Double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  double x(2.5); 
  double y(1.7);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.0, a.d_.val());

  AVEC w = createAVEC(z.val_);
  VEC g;
  a.d_.grad(w,g);
  EXPECT_FLOAT_EQ(0,g[0]);
}
TEST(AgradFwdFma,Double_FvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;  

  double x(2.5);
  fvar<var> y(1.7,1.5);
  double z(1.5);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(2.5 * 1.5, a.d_.val());

  AVEC w = createAVEC(y.val_);
  VEC g;
  a.d_.grad(w,g);
  EXPECT_FLOAT_EQ(0,g[0]);
}


TEST(AgradFwdFma,FvarFvarVar_FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z;
  z.val_.val_ = 1.7;

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5, a.val_.d_.val());
  EXPECT_FLOAT_EQ(2.5, a.d_.val_.val());
  EXPECT_FLOAT_EQ(1, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_,y.val_.val_,z.val_.val_);
  VEC r;
  a.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.5, r[0]);
  EXPECT_FLOAT_EQ(2.5, r[1]);
  EXPECT_FLOAT_EQ(1, r[2]);
}

TEST(AgradFwdFma,FvarFvarVar_Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  double y(1.5);

  fvar<fvar<var> > z;
  z.val_.val_ = 1.7;

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_,z.val_.val_);
  VEC r;
  a.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.5, r[0]);
  EXPECT_FLOAT_EQ(1, r[1]);
}

TEST(AgradFwdFma,FvarFvarVar_FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  double z(1.7);

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5, a.val_.d_.val());
  EXPECT_FLOAT_EQ(2.5, a.d_.val_.val());
  EXPECT_FLOAT_EQ(1, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_,y.val_.val_);
  VEC r;
  a.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.5, r[0]);
  EXPECT_FLOAT_EQ(2.5, r[1]);
}

TEST(AgradFwdFma,FvarFvarVar_double_double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  double y(1.5);
  double z(1.7);

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  a.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.5, r[0]);
}

TEST(AgradFwdFma,Double_FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(2.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z;
  z.val_.val_ = 1.7;

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(2.5, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_,z.val_.val_);
  VEC r;
  a.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.5, r[0]);
  EXPECT_FLOAT_EQ(1, r[1]);
}

TEST(AgradFwdFma,Double_Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(2.5);
  double y(1.5);

  fvar<fvar<var> > z;
  z.val_.val_ = 1.7;

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(z.val_.val_);
  VEC r;
  a.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1, r[0]);
}

TEST(AgradFwdFma,Double_FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(2.5);
  double z(1.7);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(2.5, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  a.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.5, r[0]);
}

TEST(AgradFwdFma,FvarFvarVar_FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z;
  z.val_.val_ = 1.7;

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5, a.val_.d_.val());
  EXPECT_FLOAT_EQ(2.5, a.d_.val_.val());
  EXPECT_FLOAT_EQ(1, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_,y.val_.val_,z.val_.val_);
  VEC r;
  a.val_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
  EXPECT_FLOAT_EQ(1, r[1]);
  EXPECT_FLOAT_EQ(0, r[2]);
}

TEST(AgradFwdFma,FvarFvarVar_FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z;
  z.val_.val_ = 1.7;

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5, a.val_.d_.val());
  EXPECT_FLOAT_EQ(2.5, a.d_.val_.val());
  EXPECT_FLOAT_EQ(1, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_,y.val_.val_,z.val_.val_);
  VEC r;
  a.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1, r[0]);
  EXPECT_FLOAT_EQ(0, r[1]);
  EXPECT_FLOAT_EQ(0, r[2]);
}
TEST(AgradFwdFma,FvarFvarVar_Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  double y(1.5);

  fvar<fvar<var> > z;
  z.val_.val_ = 1.7;

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_,z.val_.val_);
  VEC r;
  a.val_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
  EXPECT_FLOAT_EQ(0, r[1]);
}

TEST(AgradFwdFma,FvarFvarVar_FvarFvarVar_Double_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  double z(1.7);

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5, a.val_.d_.val());
  EXPECT_FLOAT_EQ(2.5, a.d_.val_.val());
  EXPECT_FLOAT_EQ(1, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_,y.val_.val_);
  VEC r;
  a.val_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
  EXPECT_FLOAT_EQ(1, r[1]);
}
TEST(AgradFwdFma,FvarFvarVar_FvarFvarVar_Double_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  double z(1.7);

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5, a.val_.d_.val());
  EXPECT_FLOAT_EQ(2.5, a.d_.val_.val());
  EXPECT_FLOAT_EQ(1, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_,y.val_.val_);
  VEC r;
  a.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1, r[0]);
  EXPECT_FLOAT_EQ(0, r[1]);
}
TEST(AgradFwdFma,FvarFvarVar_double_double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  double y(1.5);
  double z(1.7);

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  a.val_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}

TEST(AgradFwdFma,Double_FvarFvarVar_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(2.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z;
  z.val_.val_ = 1.7;

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(2.5, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_,z.val_.val_);
  VEC r;
  a.val_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
  EXPECT_FLOAT_EQ(0, r[1]);
}

TEST(AgradFwdFma,Double_Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(2.5);
  double y(1.5);

  fvar<fvar<var> > z;
  z.val_.val_ = 1.7;

  fvar<fvar<var> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(z.val_.val_);
  VEC r;
  a.val_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}
TEST(AgradFwdFma,FvarFvarVar_FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z;
  z.val_.val_ = 1.7;

  fvar<fvar<var> > a = fma(x,y,z);

  AVEC q = createAVEC(x.val_.val_,y.val_.val_,z.val_.val_);
  VEC r;
  a.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
  EXPECT_FLOAT_EQ(0, r[1]);
  EXPECT_FLOAT_EQ(0, r[2]);
}
TEST(AgradFwdFma,FvarFvarVar_Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  double y(1.5);

  fvar<fvar<var> > z;
  z.val_.val_ = 1.7;

  fvar<fvar<var> > a = fma(x,y,z);

  AVEC q = createAVEC(x.val_.val_,z.val_.val_);
  VEC r;
  a.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
  EXPECT_FLOAT_EQ(0, r[1]);
}
TEST(AgradFwdFma,FvarFvarVar_FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  double z(1.7);

  fvar<fvar<var> > a = fma(x,y,z);

  AVEC q = createAVEC(x.val_.val_,y.val_.val_);
  VEC r;
  a.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
  EXPECT_FLOAT_EQ(0, r[1]);
}
TEST(AgradFwdFma,FvarFvarVar_double_double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  double y(1.5);
  double z(1.7);

  fvar<fvar<var> > a = fma(x,y,z);

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  a.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}

TEST(AgradFwdFma,Double_FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(2.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > z;
  z.val_.val_ = 1.7;
  z.d_.val_ = 1.0;
  z.val_.d_ = 1.0;

  fvar<fvar<var> > a = fma(x,y,z);

  AVEC q = createAVEC(y.val_.val_,z.val_.val_);
  VEC r;
  a.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
  EXPECT_FLOAT_EQ(0, r[1]);
}

TEST(AgradFwdFma,Double_Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(2.5);
  double y(1.5);

  fvar<fvar<var> > z;
  z.val_.val_ = 1.7;
  z.val_.d_ = 1.0;
  z.d_.val_ = 1.0;

  fvar<fvar<var> > a = fma(x,y,z);

  AVEC q = createAVEC(z.val_.val_);
  VEC r;
  a.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}

TEST(AgradFwdFma,Double_FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(2.5);
  double z(1.7);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = fma(x,y,z);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  a.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}

struct fma_fun {
  template <typename T0, typename T1, typename T2>
  inline
  typename stan::return_type<T0,T1,T2>::type
  operator()(const T0& arg1,
             const T1& arg2,
             const T2& arg3) const {
    return fma(arg1,arg2,arg3);
  }
};

TEST(AgradFwdFma,fma_NaN) {
  fma_fun fma_;
  test_nan_mix(fma_,0.6,0.3,0.5,false);
}
