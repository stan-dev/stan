#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/exp2.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/exp2.hpp>

class AgradFwdExp2 : public testing::Test {
  void SetUp() {
  }
};


TEST_F(AgradFwdExp2,Fvar) {
  using stan::math::fvar;
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


TEST_F(AgradFwdExp2,FvarFvarDouble) {
  using stan::math::fvar;
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

struct exp2_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return exp2(arg1);
  }
};

TEST_F(AgradFwdExp2,exp2_NaN) {
  exp2_fun exp2_;
  test_nan_fwd(exp2_,false);
}
