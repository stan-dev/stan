#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/logit.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/logit.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>

class AgradFwdLogit : public testing::Test {
  void SetUp() {
    stan::math::recover_memory();
  }
};

TEST_F(AgradFwdLogit,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::logit;

  fvar<var> x(0.5,1.3);
  fvar<var> a = logit(x);

  EXPECT_FLOAT_EQ(logit(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / (0.5 - 0.25), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1 / (0.5 - 0.25), g[0]);
}
TEST_F(AgradFwdLogit,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::logit;

  fvar<var> x(0.5,1.3);
  fvar<var> a = logit(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST_F(AgradFwdLogit,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::logit;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = logit(x);

  EXPECT_FLOAT_EQ(logit(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1 / (0.5 - 0.25), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  stan::math::recover_memory();
  EXPECT_FLOAT_EQ(1 / (0.5 - 0.25), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = logit(y);
  EXPECT_FLOAT_EQ(logit(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(1 / (0.5 - 0.25), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1 / (0.5 - 0.25), r[0]);
}
TEST_F(AgradFwdLogit,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::logit;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = logit(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  stan::math::recover_memory();
  EXPECT_FLOAT_EQ(0, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = logit(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}
TEST_F(AgradFwdLogit,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = logit(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(32, g[0]);
}

struct logit_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return logit(arg1);
  }
};

TEST_F(AgradFwdLogit,logit_NaN) {
  logit_fun logit_;
  test_nan_mix(logit_,false);
}
