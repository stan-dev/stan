#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdAtan2,Fvar) {
  using stan::agrad::fvar;
  using std::atan2;

  fvar<double> x(0.5,1.0);
  fvar<double> y(2.3,2.0);
  double w = 2.1;

  fvar<double> a = atan2(x, y);
  EXPECT_FLOAT_EQ(atan2(0.5, 2.3), a.val_);
  EXPECT_FLOAT_EQ((1.0 * 2.3 - 0.5 * 2.0) / (0.5 * 0.5 + 2.3 * 2.3), a.d_);

  fvar<double> b = atan2(w, x);
  EXPECT_FLOAT_EQ(atan2(2.1, 0.5), b.val_);
  EXPECT_FLOAT_EQ((-2.1 * 1.0) / (2.1 * 2.1 + 0.5 * 0.5), b.d_);

  fvar<double> c = atan2(x, w);
  EXPECT_FLOAT_EQ(atan2(0.5, 2.1), c.val_);
  EXPECT_FLOAT_EQ((1.0 * 2.1) / (0.5 * 0.5 + 2.1 * 2.1), c.d_);
}

TEST(AgradFwdAtan2,FvarVar_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  fvar<var> x(1.5,1.3);
  fvar<var> z(1.5,1.0);
  fvar<var> a = atan2(x,z);

  EXPECT_FLOAT_EQ(atan2(1.5,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(0.1, a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / 3.0, g[0]);
  EXPECT_FLOAT_EQ(-1.0 / 3.0,g[1]);
}

TEST(AgradFwdAtan2,FvarVar_double_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  fvar<var> x(1.5,1.3);
  double z(1.5);
  fvar<var> a = atan2(x,z);

  EXPECT_FLOAT_EQ(atan2(1.5,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(13.0 / 30.0 , a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / 3.0, g[0]);
}

TEST(AgradFwdAtan2,Double_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  double x(1.5);
  fvar<var> z(1.5,1.0);
  fvar<var> a = atan2(x,z);

  EXPECT_FLOAT_EQ(atan2(1.5,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(-1.0 / 3.0, a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.0 / 3.0,g[0]);
}

TEST(AgradFwdAtan2,FvarVar_FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  fvar<var> x(1.5,1.3);
  fvar<var> z(1.5,1.0);
  fvar<var> a = atan2(x,z);

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-13.0 / 45.0, g[0]);
  EXPECT_FLOAT_EQ(2.0 / 9.0,g[1]);
}

TEST(AgradFwdAtan2,FvarVar_double_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  fvar<var> x(1.5,1.3);
  double z(1.5);
  fvar<var> a = atan2(x,z);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-13.0 / 45.0, g[0]);
}

TEST(AgradFwdAtan2,Double_FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  double x(1.5);
  fvar<var> z(1.5,1.0);
  fvar<var> a = atan2(x,z);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(2.0 / 9.0,g[0]);
}

TEST(AgradFwdAtan2,FvarFvarDouble) {
  using stan::agrad::fvar;
  using std::atan2;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  double z = 1.5;

  fvar<fvar<double> > a = atan2(x,y);

  EXPECT_FLOAT_EQ(atan(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(-1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.d_.val_);
  EXPECT_FLOAT_EQ((1.5 * 1.5 - 1.5 * 1.5) / ((1.5 * 1.5 + 1.5 * 1.5) * (1.5 * 1.5 + 1.5 * 1.5)), a.d_.d_);

  a = atan2(x,z);
  EXPECT_FLOAT_EQ(atan(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0.0, a.d_.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_.d_);

  a = atan2(z, y);

  EXPECT_FLOAT_EQ(atan(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0.0, a.val_.d_);
  EXPECT_FLOAT_EQ(-1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_.d_);
}

TEST(AgradFwdAtan2,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = atan2(x,y);

  EXPECT_FLOAT_EQ(atan(1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(-1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.d_.val_.val());
  EXPECT_FLOAT_EQ((1.5 * 1.5 - 1.5 * 1.5) / ((1.5 * 1.5 + 1.5 * 1.5) * (1.5 * 1.5 + 1.5 * 1.5)), a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(q,g);
  EXPECT_FLOAT_EQ(1.0 / 3.0, g[0]);
  EXPECT_FLOAT_EQ(-1.0 / 3.0,g[1]);
}

TEST(AgradFwdAtan2,FvarFvarVar_Double_1stDeriv) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  using std::atan2;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;

  double y (1.5);

  fvar<fvar<var> > a = atan2(x,y);

  EXPECT_FLOAT_EQ(atan(1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0.0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0.0, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(q,g);
  EXPECT_FLOAT_EQ(1.0 / 3.0, g[0]);
}


TEST(AgradFwdAtan2,Double_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  double x(1.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = atan2(x,y);

  EXPECT_FLOAT_EQ(atan(1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(-1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0.0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(q,g);
  EXPECT_FLOAT_EQ(-1.0 / 3.0,g[0]);
}

TEST(AgradFwdAtan2,FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = atan2(x,y);

  AVEC q = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(q,g);
  EXPECT_FLOAT_EQ(-2.0 / 9.0, g[0]);
}
TEST(AgradFwdAtan2,FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = atan2(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC h;
  a.d_.val_.grad(p,h);
  EXPECT_FLOAT_EQ(2.0 / 9.0, h[0]);
}
TEST(AgradFwdAtan2,FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = atan2(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC h;
  a.d_.d_.grad(p,h);
  EXPECT_FLOAT_EQ(8.0 / 27.0, h[0]);
  EXPECT_FLOAT_EQ(-8.0 / 27.0, h[1]);
}

TEST(AgradFwdAtan2,FvarFvarVar_Double_2ndDeriv) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  using std::atan2;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;

  double y (1.5);

  fvar<fvar<var> > a = atan2(x,y);

  AVEC q = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(q,g);
  EXPECT_FLOAT_EQ(-2.0 / 9.0, g[0]);
}
TEST(AgradFwdAtan2,FvarFvarVar_Double_3rdDeriv) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  using std::atan2;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  double y (1.5);

  fvar<fvar<var> > a = atan2(x,y);

  AVEC q = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(q,g);
  EXPECT_FLOAT_EQ(4.0 / 27.0, g[0]);
}


TEST(AgradFwdAtan2,Double_FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  double x(1.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = atan2(x,y);

  AVEC q = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(q,g);
  EXPECT_FLOAT_EQ(2.0 / 9.0,g[0]);
}
TEST(AgradFwdAtan2,Double_FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  double x(1.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = atan2(x,y);

  AVEC q = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(q,g);
  EXPECT_FLOAT_EQ(-4.0 / 27.0,g[0]);
}


struct atan2_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return atan2(arg1,arg2);
  }
};

TEST(AgradFwdAtan2, nan) {
  atan2_fun atan2_;
  test_nan(atan2_,3.0,5.0,false);
}
