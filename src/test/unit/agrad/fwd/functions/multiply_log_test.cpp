#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdMultiplyLog,Fvar) {
  using stan::agrad::fvar;
  using std::isnan;
  using std::log;
  using stan::math::multiply_log;

  fvar<double> x(0.5,1.0);
  fvar<double> y(1.2,2.0);
  fvar<double> z(-0.4,3.0);

  double w = 0.0;
  double v = 1.3;

  fvar<double> a = multiply_log(x, y);
  EXPECT_FLOAT_EQ(multiply_log(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ(1.0 * log(1.2) + 0.5 * 2.0 / 1.2, a.d_);

  fvar<double> b = multiply_log(x,z);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> c = multiply_log(x, v);
  EXPECT_FLOAT_EQ(multiply_log(0.5, 1.3), c.val_);
  EXPECT_FLOAT_EQ(log(1.3), c.d_);

  fvar<double> d = multiply_log(v, x);
  EXPECT_FLOAT_EQ(multiply_log(1.3, 0.5), d.val_);
  EXPECT_FLOAT_EQ(1.3 * 1.0 / 0.5, d.d_);

  fvar<double> e = multiply_log(x, w);
  isnan(e.val_);
  isnan(e.d_);
}
TEST(AgradFwdMultiplyLog,FvarVar_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  fvar<var> x(1.5,1.3);
  fvar<var> z(1.8,1.1);
  fvar<var> a = multiply_log(x,z);

  EXPECT_FLOAT_EQ(multiply_log(1.5,1.8), a.val_.val());
  EXPECT_FLOAT_EQ(log(1.8) * 1.3 + 1.5 * 1.1 / 1.8, a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(log(1.8), g[0]);
  EXPECT_FLOAT_EQ(1.5 / 1.8, g[1]);
}
TEST(AgradFwdMultiplyLog,FvarVar_Double_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  fvar<var> x(1.5,1.3);
  double z(1.8);
  fvar<var> a = multiply_log(x,z);

  EXPECT_FLOAT_EQ(multiply_log(1.5,1.8), a.val_.val());
  EXPECT_FLOAT_EQ(log(1.8) * 1.3, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(log(1.8), g[0]);
}
TEST(AgradFwdMultiplyLog,Double_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  double x(1.5);
  fvar<var> z(1.8,1.1);
  fvar<var> a = multiply_log(x,z);

  EXPECT_FLOAT_EQ(multiply_log(1.5,1.8), a.val_.val());
  EXPECT_FLOAT_EQ(1.5 * 1.1 / 1.8, a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.5 / 1.8, g[0]);
}
TEST(AgradFwdMultiplyLog,FvarVar_FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  fvar<var> x(1.5,1.3);
  fvar<var> z(1.8,1.1);
  fvar<var> a = multiply_log(x,z);

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.1 / 1.8, g[0]);
  EXPECT_FLOAT_EQ(1.3 / 1.8 - 1.5 * 1.1 / 1.8 / 1.8, g[1]);
}
TEST(AgradFwdMultiplyLog,FvarVar_Double_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  fvar<var> x(1.5,1.3);
  double z(1.8);
  fvar<var> a = multiply_log(x,z);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdMultiplyLog,Double_FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  double x(1.5);
  fvar<var> z(1.8,1.1);
  fvar<var> a = multiply_log(x,z);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.5 * 1.1 / 1.8 / 1.8, g[0]);
}
TEST(AgradFwdMultiplyLog,FvarFvarDouble) {
  using stan::agrad::fvar;
  using std::log;
  using stan::math::multiply_log;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.3;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.8;
  y.d_.val_ = 1.1;

  fvar<fvar<double> > a = multiply_log(x,y);

  EXPECT_FLOAT_EQ(multiply_log(1.5,1.8), a.val_.val_);
  EXPECT_FLOAT_EQ(log(1.8) * 1.3, a.val_.d_);
  EXPECT_FLOAT_EQ(1.5 / 1.8 * 1.1, a.d_.val_);
  EXPECT_FLOAT_EQ(143.0 / 180.0, a.d_.d_);
}
TEST(AgradFwdMultiplyLog,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.3;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.8;
  y.d_.val_ = 1.1;

  fvar<fvar<var> > a = multiply_log(x,y);

  EXPECT_FLOAT_EQ(multiply_log(1.5,1.8), a.val_.val_.val());
  EXPECT_FLOAT_EQ(log(1.8) * 1.3, a.val_.d_.val());
  EXPECT_FLOAT_EQ(1.5 / 1.8 * 1.1, a.d_.val_.val());
  EXPECT_FLOAT_EQ(143.0 / 180.0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(log(1.8), g[0]);
  EXPECT_FLOAT_EQ(1.5 / 1.8, g[1]);
}
TEST(AgradFwdMultiplyLog,FvarFvarVar_Double_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.3;
  double y(1.8);

  fvar<fvar<var> > a = multiply_log(x,y);

  EXPECT_FLOAT_EQ(multiply_log(1.5,1.8), a.val_.val_.val());
  EXPECT_FLOAT_EQ(log(1.8) * 1.3, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(log(1.8), g[0]);
}
TEST(AgradFwdMultiplyLog,Double_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  double x(1.5);
  fvar<fvar<var> > y;
  y.val_.val_ = 1.8;
  y.d_.val_ = 1.1;

  fvar<fvar<var> > a = multiply_log(x,y);

  EXPECT_FLOAT_EQ(multiply_log(1.5,1.8), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(1.5 / 1.8 * 1.1, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.5 / 1.8, g[0]);
}
TEST(AgradFwdMultiplyLog,FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.3;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.8;
  y.d_.val_ = 1.1;

  fvar<fvar<var> > a = multiply_log(x,y);

  EXPECT_FLOAT_EQ(multiply_log(1.5,1.8), a.val_.val_.val());
  EXPECT_FLOAT_EQ(log(1.8) * 1.3, a.val_.d_.val());
  EXPECT_FLOAT_EQ(1.5 / 1.8 * 1.1, a.d_.val_.val());
  EXPECT_FLOAT_EQ(143.0 / 180.0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(1.3 / 1.8, g[1]);
}
TEST(AgradFwdMultiplyLog,FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.3;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.8;
  y.d_.val_ = 1.1;

  fvar<fvar<var> > a = multiply_log(x,y);

  EXPECT_FLOAT_EQ(multiply_log(1.5,1.8), a.val_.val_.val());
  EXPECT_FLOAT_EQ(log(1.8) * 1.3, a.val_.d_.val());
  EXPECT_FLOAT_EQ(1.5 / 1.8 * 1.1, a.d_.val_.val());
  EXPECT_FLOAT_EQ(143.0 / 180.0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.1 / 1.8, g[0]);
  EXPECT_FLOAT_EQ(1.5 * -1.1 / 1.8 / 1.8, g[1]);
}
TEST(AgradFwdMultiplyLog,FvarFvarVar_Double_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.3;
  double y(1.8);

  fvar<fvar<var> > a = multiply_log(x,y);

  EXPECT_FLOAT_EQ(multiply_log(1.5,1.8), a.val_.val_.val());
  EXPECT_FLOAT_EQ(log(1.8) * 1.3, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdMultiplyLog,Double_FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  double x(1.5);
  fvar<fvar<var> > y;
  y.val_.val_ = 1.8;
  y.d_.val_ = 1.1;

  fvar<fvar<var> > a = multiply_log(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-1.5 / 1.8 / 1.8 * 1.1, g[0]);
}
TEST(AgradFwdMultiplyLog,FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.3;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.8;
  y.d_.val_ = 1.1;

  fvar<fvar<var> > a = multiply_log(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(-0.44135803, g[1]);
}
TEST(AgradFwdMultiplyLog,FvarFvarVar_Double_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;
  double y(1.8);

  fvar<fvar<var> > a = multiply_log(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdMultiplyLog,Double_FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using stan::math::multiply_log;

  double x(1.5);
  fvar<fvar<var> > y;
  y.val_.val_ = 1.8;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = multiply_log(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.51440328, g[0]);
}

struct multiply_log_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return multiply_log(arg1,arg2);
  }
};

TEST(AgradFwdMultiplyLog, nan) {
  multiply_log_fun multiply_log_;
  test_nan(multiply_log_,3.0,5.0,false);
}
