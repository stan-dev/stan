#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/fmax.hpp>
#include <stan/math/rev/scal/fun/fmax.hpp>


TEST(AgradFwdFmax,FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(2.5,1.3);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fmax(x,z);

  EXPECT_FLOAT_EQ(fmax(2.5,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3, a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1,g[0]);
  std::isnan(g[1]);
}
TEST(AgradFwdFmax,FvarVar_double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(2.5,1.3);
  double z(1.5);
  fvar<var> a = fmax(x,z);

  EXPECT_FLOAT_EQ(fmax(2.5,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1,g[0]);
}
TEST(AgradFwdFmax,Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(2.5);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fmax(x,z);

  EXPECT_FLOAT_EQ(fmax(2.5,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  std::isnan(g[0]);
}
TEST(AgradFwdFmax,FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(2.5,1.3);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fmax(x,z);

  EXPECT_FLOAT_EQ(fmax(2.5,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3, a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0,g[0]);
  std::isnan(g[1]);
}
TEST(AgradFwdFmax,FvarVar_double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(2.5,1.3);
  double z(1.5);
  fvar<var> a = fmax(x,z);

  EXPECT_FLOAT_EQ(fmax(2.5,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0,g[0]);
}
TEST(AgradFwdFmax,Double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(2.5);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fmax(x,z);

  EXPECT_FLOAT_EQ(fmax(2.5,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  std::isnan(g[0]);
}

TEST(AgradFwdFmax,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fmax(x,y);

  EXPECT_FLOAT_EQ(fmax(2.5,1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_,y.val_.val_);
  VEC r;
  a.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.0, r[0]);
  EXPECT_FLOAT_EQ(0.0, r[1]);
}
TEST(AgradFwdFmax,FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  double y(1.5);

  fvar<fvar<var> > a = fmax(x,y);

  EXPECT_FLOAT_EQ(fmax(2.5,1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  a.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.0, r[0]);
}
TEST(AgradFwdFmax,Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(2.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fmax(x,y);

  EXPECT_FLOAT_EQ(fmax(2.5,1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  a.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0.0, r[0]);
}
TEST(AgradFwdFmax,FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fmax(x,y);

  AVEC q = createAVEC(x.val_.val_,y.val_.val_);
  VEC r;
  a.val_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0.0, r[0]);
  EXPECT_FLOAT_EQ(0.0, r[1]);
}
TEST(AgradFwdFmax,FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fmax(x,y);

  AVEC q = createAVEC(x.val_.val_,y.val_.val_);
  VEC r;
  a.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0.0, r[0]);
  EXPECT_FLOAT_EQ(0.0, r[1]);
}
TEST(AgradFwdFmax,FvarFvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  double y(1.5);

  fvar<fvar<var> > a = fmax(x,y);

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  a.val_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0.0, r[0]);
}

TEST(AgradFwdFmax,Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(2.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fmax(x,y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  a.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0.0, r[0]);
}
TEST(AgradFwdFmax,FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fmax(x,y);

  AVEC q = createAVEC(x.val_.val_,y.val_.val_);
  VEC r;
  a.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0.0, r[0]);
  EXPECT_FLOAT_EQ(0.0, r[1]);
}
TEST(AgradFwdFmax,FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  double y(1.5);

  fvar<fvar<var> > a = fmax(x,y);

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  a.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0.0, r[0]);
}

TEST(AgradFwdFmax,Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(2.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = fmax(x,y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  a.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0.0, r[0]);
}

struct fmax_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return fmax(arg1,arg2);
  }
};

TEST(AgradFwdFmax, nan) {
  fmax_fun fmax_;
  double nan = std::numeric_limits<double>::quiet_NaN();
  test_nan_mix(fmax_,nan,nan,false);
}
