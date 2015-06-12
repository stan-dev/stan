#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/exp2.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/exp2.hpp>
#include <stan/math/rev/scal/fun/exp2.hpp>

class AgradFwdExp2 : public testing::Test {
  void SetUp() {
    stan::math::recover_memory();
  }
};




TEST_F(AgradFwdExp2,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::exp2;
  using std::log;

  fvar<var> x(0.5,1.3);
  fvar<var> a = exp2(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * exp2(0.5) * log(2) * log(2), g[0]);
}


TEST_F(AgradFwdExp2,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
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
  stan::math::recover_memory();
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
  using stan::math::fvar;
  using stan::math::var;
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
  stan::math::recover_memory();

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
  using stan::math::fvar;
  using stan::math::var;
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
  test_nan_mix(exp2_,false);
}
