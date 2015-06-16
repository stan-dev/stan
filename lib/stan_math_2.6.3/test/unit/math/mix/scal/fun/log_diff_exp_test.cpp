#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log_diff_exp.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/log_diff_exp.hpp>
#include <stan/math/rev/scal/fun/log_diff_exp.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>

TEST(AgradFwdLogDiffExp,FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
TEST(AgradFwdLogDiffExp,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  test_nan_mix(log_diff_exp_,3.0,5.0,false);
}
