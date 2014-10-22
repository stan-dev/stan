#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/math/functions/log_diff_exp.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdLogDiffExp,Fvar) {
  using stan::agrad::fvar;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<double> x(1.2);
  fvar<double> y(0.5);
  x.d_ = 1.0;
  y.d_ = 2.0;

  double z = 1.1;

  fvar<double> a = log_diff_exp(x, y);
  EXPECT_FLOAT_EQ(log_diff_exp(1.2, 0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 - exp(0.5 - 1.2) ) + 2 / (1 - exp(1.2 - 0.5) ), a.d_);

  fvar<double> b = log_diff_exp(x, z);
  EXPECT_FLOAT_EQ(log_diff_exp(1.2, 1.1), b.val_);
  EXPECT_FLOAT_EQ(1 / (1 - exp(1.1 - 1.2) ), b.d_);

  fvar<double> c = log_diff_exp(z, y);
  EXPECT_FLOAT_EQ(log_diff_exp(1.1, 0.5), c.val_);
  EXPECT_FLOAT_EQ(2 / (1 - exp(1.1 - 0.5) ), c.d_);
}

TEST(AgradFwdLogDiffExp, AgradFvar_exception) {
  using stan::agrad::fvar;
  EXPECT_NO_THROW(log_diff_exp(fvar<double>(3), fvar<double>(4)));
  EXPECT_NO_THROW(log_diff_exp(fvar<double>(3), 4));
  EXPECT_NO_THROW(log_diff_exp(3, fvar<double>(4)));
}

TEST(AgradFwdLogDiffExp,FvarVar_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<var> x(9.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_diff_exp(x,z);

  EXPECT_FLOAT_EQ(log_diff_exp(9.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.3 * exp(9.0) - 1.0 * exp(6.0)) / (exp(9.0) - exp(6.0)), a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(9.0) / (exp(9.0) - exp(6.0)),g[0]);
  EXPECT_FLOAT_EQ(-exp(6.0) / (exp(9.0) - exp(6.0)),g[1]);
}
TEST(AgradFwdLogDiffExp,FvarVar_Double_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<var> x(9.0,1.3);
  double z(6.0);
  fvar<var> a = log_diff_exp(x,z);

  EXPECT_FLOAT_EQ(log_diff_exp(9.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.3 * exp(9.0)) / (exp(9.0) - exp(6.0)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(9.0) / (exp(9.0) - exp(6.0)),g[0]);
}
TEST(AgradFwdLogDiffExp,Double_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  double x(9.0);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_diff_exp(x,z);

  EXPECT_FLOAT_EQ(log_diff_exp(9.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((-1.0 * exp(6.0)) / (exp(9.0) - exp(6.0)), a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-exp(6.0) / (exp(9.0) - exp(6.0)),g[0]);
}
TEST(AgradFwdLogDiffExp,FvarVar_FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<var> x(9.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_diff_exp(x,z);

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ((1.3 * exp(9.0) * (exp(9.0) - exp(6.0)) - exp(9.0) 
                   * (1.3 * exp(9.0) - exp(6.0))) / (exp(9.0) - exp(6.0)) 
                  / (exp(9.0) - exp(6.0)) ,g[0]);
  EXPECT_FLOAT_EQ((-exp(6.0) * (exp(9.0) - exp(6.0)) + exp(6.0) 
                   * (1.3 * exp(9.0) - exp(6.0))) / (exp(9.0) - exp(6.0)) 
                  / (exp(9.0) - exp(6.0)) ,g[1]);}
TEST(AgradFwdLogDiffExp,FvarVar_Double_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<var> x(9.0,1.3);
  double z(6.0);
  fvar<var> a = log_diff_exp(x,z);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ((1.3 * exp(9.0) * (exp(9.0) - exp(6.0)) - exp(9.0) * 1.3 
                   * exp(9.0)) / (exp(9.0) - exp(6.0)) / (exp(9.0) - exp(6.0))
                  ,g[0]);
}
TEST(AgradFwdLogDiffExp,Double_FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  double x(9.0);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_diff_exp(x,z);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ((-exp(6.0) * (exp(9.0) - exp(6.0)) + exp(6.0) * -exp(6.0))
                  / (exp(9.0) - exp(6.0)) / (exp(9.0) - exp(6.0)),g[0]);
}
TEST(AgradFwdLogDiffExp,FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = 9.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = log_diff_exp(x,y);

  EXPECT_FLOAT_EQ(log_diff_exp(9.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(9.0) / (exp(9.0) - exp(6.0)), a.val_.d_);
  EXPECT_FLOAT_EQ(-exp(6.0) / (exp(9.0) - exp(6.0)), a.d_.val_);
  EXPECT_FLOAT_EQ(0.055141006, a.d_.d_);
}
TEST(AgradFwdLogDiffExp,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 9.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_diff_exp(x,y);

  EXPECT_FLOAT_EQ(log_diff_exp(9.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(9.0) / (exp(9.0) - exp(6.0)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(-exp(6.0) / (exp(9.0) - exp(6.0)), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0.055141006, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(9.0) / (exp(9.0) - exp(6.0)), g[0]);
  EXPECT_FLOAT_EQ(-exp(6.0) / (exp(9.0) - exp(6.0)), g[1]);
}
TEST(AgradFwdLogDiffExp,FvarFvarVar_Double_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 9.0;
  x.val_.d_ = 1.0;

  double y(6.0);

  fvar<fvar<var> > a = log_diff_exp(x,y);

  EXPECT_FLOAT_EQ(log_diff_exp(9.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(9.0) / (exp(9.0) - exp(6.0)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(9.0) / (exp(9.0) - exp(6.0)), g[0]);
}

TEST(AgradFwdLogDiffExp,Double_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  double x(9.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_diff_exp(x,y);

  EXPECT_FLOAT_EQ(log_diff_exp(9.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(-exp(6.0) / (exp(9.0) - exp(6.0)), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-exp(6.0) / (exp(9.0) - exp(6.0)), g[0]);
}
TEST(AgradFwdLogDiffExp,FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 9.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_diff_exp(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ((exp(9.0) * (exp(9.0) - exp(6.0)) - exp(9.0) 
                   * exp(9.0)) / (exp(9.0) - exp(6.0)) / (exp(9.0) - exp(6.0))
                  ,g[0]);
  EXPECT_FLOAT_EQ(0.055141006, g[1]);
}
TEST(AgradFwdLogDiffExp,FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 9.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_diff_exp(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.055141006, g[0]);
  EXPECT_FLOAT_EQ((-exp(6.0) * (exp(9.0) - exp(6.0)) + exp(6.0) * -exp(6.0))
                  / (exp(9.0) - exp(6.0)) / (exp(9.0) - exp(6.0)),g[1]);
}
TEST(AgradFwdLogDiffExp,FvarFvarVar_Double_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 9.0;
  x.val_.d_ = 1.0;

  double y(6.0);

  fvar<fvar<var> > a = log_diff_exp(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ((exp(9.0) * (exp(9.0) - exp(6.0)) - exp(9.0) 
                   * exp(9.0)) / (exp(9.0) - exp(6.0)) / (exp(9.0) - exp(6.0))
                  ,g[0]);
}

TEST(AgradFwdLogDiffExp,Double_FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  double x(9.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_diff_exp(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ((-exp(6.0) * (exp(9.0) - exp(6.0)) + exp(6.0) * -exp(6.0))
                  / (exp(9.0) - exp(6.0)) / (exp(9.0) - exp(6.0)),g[0]);
}
TEST(AgradFwdLogDiffExp,FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 9.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_diff_exp(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.060919307, g[0]);
  EXPECT_FLOAT_EQ(0.060919307,g[1]);
}
TEST(AgradFwdLogDiffExp,FvarFvarVar_Double_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 9.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  double y(6.0);

  fvar<fvar<var> > a = log_diff_exp(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.060919308279076959008006310,g[0]);
}

TEST(AgradFwdLogDiffExp,Double_FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  double x(9.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = log_diff_exp(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.060919308279076959008006309952,g[0]);
}

struct log_diff_exp_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return log_diff_exp(arg1,arg2);
  }
};

TEST(AgradFwdLogDiffExp, nan) {
  log_diff_exp_fun log_diff_exp_;
  test_nan(log_diff_exp_,3.0,5.0,false);
}
