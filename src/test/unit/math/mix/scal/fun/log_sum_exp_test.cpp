#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log_sum_exp.hpp>
#include <stan/math/prim/arr/fun/log_sum_exp.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/arr/fun/log_sum_exp.hpp>
#include <stan/math/fwd/scal/fun/log_sum_exp.hpp>
#include <stan/math/rev/scal/fun/log_sum_exp.hpp>
#include <stan/math/rev/arr/fun/log_sum_exp.hpp>
#include <stan/math/fwd/scal/fun/log1p_exp.hpp>
#include <stan/math/rev/scal/fun/log1p_exp.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>

TEST(AgradFwdLogSumExp,FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<var> x(3.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_sum_exp(x,z);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.3 * exp(3.0) + 1.0 * exp(6.0)) / (exp(3.0) + exp(6.0)), a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)),g[0]);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)),g[1]);
}
TEST(AgradFwdLogSumExp,FvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<var> x(3.0,1.3);
  double z(6.0);
  fvar<var> a = log_sum_exp(x,z);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.3 * exp(3.0)) / (exp(3.0) + exp(6.0)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)),g[0]);
}
TEST(AgradFwdLogSumExp,Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  double x(3.0);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_sum_exp(x,z);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.0 * exp(6.0)) / (exp(3.0) + exp(6.0)), a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)),g[0]);
}
TEST(AgradFwdLogSumExp,FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<var> x(3.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_sum_exp(x,z);

  EXPECT_FLOAT_EQ((1.3 * exp(3.0) + 1.0 * exp(6.0)) / (exp(3.0) + exp(6.0)), a.d_.val());


  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ((1.3 * exp(3.0) * (exp(3.0) + exp(6.0)) - exp(3.0) 
                   * (1.3* exp(3.0) + exp(6.0))) / (exp(3.0) + exp(6.0)) 
                  / (exp(3.0) + exp(6.0)),g[0]);
  EXPECT_FLOAT_EQ((exp(6.0) * (exp(3.0) + exp(6.0)) - exp(6.0) 
                   * (1.3* exp(3.0) + exp(6.0))) / (exp(3.0) + exp(6.0)) 
                  / (exp(3.0) + exp(6.0)),g[1]);
}
TEST(AgradFwdLogSumExp,FvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<var> x(3.0,1.3);
  double z(6.0);
  fvar<var> a = log_sum_exp(x,z);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * (exp(3.0) * (exp(3.0) + exp(6.0)) - exp(3.0) * exp(3.0))
                  / (exp(3.0) + exp(6.0)) / (exp(3.0) + exp(6.0)),g[0]);
}
TEST(AgradFwdLogSumExp,Double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  double x(3.0);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_sum_exp(x,z);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ((exp(6.0) * (exp(3.0) + exp(6.0)) - exp(6.0) * exp(6.0))
                  / (exp(3.0) + exp(6.0)) / (exp(3.0) + exp(6.0)),g[0]);
}
TEST(AgradFwdLogSumExp,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x,y);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), a.d_.val_.val());
  EXPECT_FLOAT_EQ(-0.045176659, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), g[0]);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), g[1]);
}
TEST(AgradFwdLogSumExp,FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  double y(6.0);

  fvar<fvar<var> > a = log_sum_exp(x,y);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), g[0]);
}
TEST(AgradFwdLogSumExp,Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x,y);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), g[0]);
}
TEST(AgradFwdLogSumExp,FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ((exp(3.0) * (exp(3.0) + exp(6.0)) - exp(3.0) 
                   * (exp(3.0))) / (exp(3.0) + exp(6.0)) 
                  / (exp(3.0) + exp(6.0)),g[0]);
  EXPECT_FLOAT_EQ(-0.045176659, g[1]);
}
TEST(AgradFwdLogSumExp,FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.045176659, g[0]);
  EXPECT_FLOAT_EQ((exp(6.0) * (exp(3.0) + exp(6.0)) - exp(6.0) 
                   * (exp(6.0))) / (exp(3.0) + exp(6.0)) 
                  / (exp(3.0) + exp(6.0)),g[1]);
}
TEST(AgradFwdLogSumExp,FvarFvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  double y(6.0);

  fvar<fvar<var> > a = log_sum_exp(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ((exp(3.0) * (exp(3.0) + exp(6.0)) - exp(3.0) 
                   * (exp(3.0))) / (exp(3.0) + exp(6.0)) 
                  / (exp(3.0) + exp(6.0)),g[0]);
}
TEST(AgradFwdLogSumExp,Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ((exp(6.0) * (exp(3.0) + exp(6.0)) - exp(6.0) 
                   * (exp(6.0))) / (exp(3.0) + exp(6.0)) 
                  / (exp(3.0) + exp(6.0)),g[0]);
}
TEST(AgradFwdLogSumExp,FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.040891573, g[0]);
  EXPECT_FLOAT_EQ(0.040891573,g[1]);
}
TEST(AgradFwdLogSumExp,FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;
  double y(6.0);

  fvar<fvar<var> > a = log_sum_exp(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.040891574660943478616430308425,g[0]);
}
TEST(AgradFwdLogSumExp,Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_sum_exp;
  using std::exp;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.040891574660943478616430308,g[0]);
}

struct log_sum_exp_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return log_sum_exp(arg1,arg2);
  }
};

TEST(AgradFwdLogSumExp, nan) {
  log_sum_exp_fun log_sum_exp_;
  test_nan_mix(log_sum_exp_,3.0,5.0,false);
}
