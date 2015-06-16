#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/inv_cloglog.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/inv_cloglog.hpp>
#include <stan/math/rev/scal/fun/inv_cloglog.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>



TEST(AgradFwdInvCLogLog,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv_cloglog;
  using std::exp;

  fvar<var> x(0.5,1.3);
  fvar<var> a = inv_cloglog(x);

  EXPECT_FLOAT_EQ(inv_cloglog(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * exp(0.5 - exp(0.5)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(0.5 - exp(0.5)), g[0]);
}
TEST(AgradFwdInvCLogLog,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv_cloglog;
  using std::exp;

  fvar<var> x(0.5,1.3);
  fvar<var> a = inv_cloglog(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-exp(0.5 - exp(0.5)) * (exp(0.5) - 1.0) * 1.3, g[0]);
}

TEST(AgradFwdInvCLogLog,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv_cloglog;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = inv_cloglog(x);

  EXPECT_FLOAT_EQ(inv_cloglog(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(0.5 - exp(0.5)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(0.5 - exp(0.5)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = inv_cloglog(y);
  EXPECT_FLOAT_EQ(inv_cloglog(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(exp(0.5 - exp(0.5)), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(exp(0.5 - exp(0.5)), r[0]);
}
TEST(AgradFwdInvCLogLog,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv_cloglog;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = inv_cloglog(x);

  EXPECT_FLOAT_EQ(inv_cloglog(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(0.5 - exp(0.5)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(0.5 - exp(0.5)) * (1.0 - exp(0.5)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = inv_cloglog(y);
  EXPECT_FLOAT_EQ(inv_cloglog(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(exp(0.5 - exp(0.5)), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(exp(0.5 - exp(0.5)) * (1.0 - exp(0.5)), r[0]);
}
TEST(AgradFwdInvCLogLog,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv_cloglog;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = inv_cloglog(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.38929006295064455041710794866, g[0]);
}

struct inv_cloglog_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return inv_cloglog(arg1);
  }
};

TEST(AgradFwdInvCLogLog,inv_cloglog_NaN) {
  inv_cloglog_fun inv_cloglog_;
  test_nan_mix(inv_cloglog_,false);
}
