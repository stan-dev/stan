#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/math/functions/logit.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

class AgradFwdLogit : public testing::Test {
  void SetUp() {
    stan::agrad::recover_memory();
  }
};


TEST_F(AgradFwdLogit,Fvar) {
  using stan::agrad::fvar;
  using stan::math::logit;
  using std::isnan;

  fvar<double> x(0.5,1.0);

  fvar<double> a = logit(x);
  EXPECT_FLOAT_EQ(logit(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 - 0.5 * 0.5), a.d_);

  fvar<double> y(-1.2,1.0);

  fvar<double> b = logit(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> z(1.5,1.0);

  fvar<double> c = logit(z);
  isnan(c.val_);
  isnan(c.d_);
}

TEST_F(AgradFwdLogit,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::logit;

  fvar<var> x(0.5,1.3);
  fvar<var> a = logit(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST_F(AgradFwdLogit,FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::logit;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = logit(x);

  EXPECT_FLOAT_EQ(logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 - 0.25), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = logit(y);
  EXPECT_FLOAT_EQ(logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(1 / (0.5 - 0.25), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST_F(AgradFwdLogit,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  stan::agrad::recover_memory();
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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::logit;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = logit(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  stan::agrad::recover_memory();
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
  using stan::agrad::fvar;
  using stan::agrad::var;

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
  test_nan(logit_,false);
}
