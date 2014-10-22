#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/math/functions/exp2.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

class AgradFwdExp2 : public testing::Test {
  void SetUp() {
    stan::agrad::recover_memory();
  }
};


TEST_F(AgradFwdExp2,Fvar) {
  using stan::agrad::fvar;
  using stan::math::exp2;
  using std::log;

  fvar<double> x(0.5,1.0);
  
  fvar<double> a = exp2(x);
  EXPECT_FLOAT_EQ(exp2(0.5), a.val_);
  EXPECT_FLOAT_EQ(exp2(0.5) * log(2), a.d_);

  fvar<double> b = 2 * exp2(x) + 4;
  EXPECT_FLOAT_EQ(2 * exp2(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 * exp2(0.5) * log(2), b.d_);

  fvar<double> c = -exp2(x) + 5;
  EXPECT_FLOAT_EQ(-exp2(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-exp2(0.5) * log(2), c.d_);

  fvar<double> d = -3 * exp2(-x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * exp2(-0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(3 * exp2(-0.5) * log(2) + 5, d.d_);

  fvar<double> y(-0.5,1.0);
  fvar<double> e = exp2(y);
  EXPECT_FLOAT_EQ(exp2(-0.5), e.val_);
  EXPECT_FLOAT_EQ(exp2(-0.5) * log(2), e.d_);

  fvar<double> z(0.0,1.0);
  fvar<double> f = exp2(z);
  EXPECT_FLOAT_EQ(exp2(0.0), f.val_);
  EXPECT_FLOAT_EQ(exp2(0.0) * log(2), f.d_);
}

TEST_F(AgradFwdExp2,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::exp2;
  using std::log;

  fvar<var> x(0.5,1.3);
  fvar<var> a = exp2(x);

  EXPECT_FLOAT_EQ(exp2(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * exp2(0.5) * log(2), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp2(0.5) * log(2), g[0]);
}

TEST_F(AgradFwdExp2,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::exp2;
  using std::log;

  fvar<var> x(0.5,1.3);
  fvar<var> a = exp2(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * exp2(0.5) * log(2) * log(2), g[0]);
}
TEST_F(AgradFwdExp2,FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::exp2;
  using std::log;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = exp2(x);

  EXPECT_FLOAT_EQ(exp2(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(exp2(0.5) * log(2), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = exp2(y);
  EXPECT_FLOAT_EQ(exp2(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(exp2(0.5) * log(2), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

TEST_F(AgradFwdExp2,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::exp2;
  using std::log;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = exp2(x);

  EXPECT_FLOAT_EQ(exp2(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp2(0.5) * log(2), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  stan::agrad::recover_memory();
  EXPECT_FLOAT_EQ(exp2(0.5) * log(2), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = exp2(y);
  EXPECT_FLOAT_EQ(exp2(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(exp2(0.5) * log(2), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(exp2(0.5) * log(2), r[0]);
}

TEST_F(AgradFwdExp2,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::exp2;
  using std::log;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = exp2(x);

  EXPECT_FLOAT_EQ(exp2(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp2(0.5) * log(2), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  stan::agrad::recover_memory();

  EXPECT_FLOAT_EQ(exp2(0.5) * log(2) * log(2), g[0]);


  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = exp2(y);
  EXPECT_FLOAT_EQ(exp2(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(exp2(0.5) * log(2), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(exp2(0.5) * log(2) * log(2), r[0]);
}
TEST_F(AgradFwdExp2,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = exp2(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(log(2)*log(2)*log(2)*exp2(0.5), g[0]);
}

struct exp2_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return exp2(arg1);
  }
};

TEST_F(AgradFwdExp2,exp2_NaN) {
  exp2_fun exp2_;
  test_nan(exp2_,false);
}
