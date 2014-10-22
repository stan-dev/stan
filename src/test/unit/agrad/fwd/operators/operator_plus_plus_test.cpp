#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdOperatorPlusPlus, Fvar) {
  using stan::agrad::fvar;

  fvar<double> x(0.5,1.0);
  x++;

  EXPECT_FLOAT_EQ(0.5 + 1.0, x.val_);
  EXPECT_FLOAT_EQ(1.0, x.d_);

  fvar<double> y(-0.5,1.0);
  y++;

  EXPECT_FLOAT_EQ(-0.5 + 1.0, y.val_);
  EXPECT_FLOAT_EQ(1.0, y.d_);
}

TEST(AgradFwdOperatorPlusPlus, FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  x++;
  EXPECT_FLOAT_EQ(0.5 + 1.0, x.val_.val());
  EXPECT_FLOAT_EQ(1.3, x.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  x.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}
TEST(AgradFwdOperatorPlusPlus, FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  x++;

  AVEC y = createAVEC(x.val_);
  VEC g;
  x.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFwdOperatorPlusPlus, FvarFvarDouble) {
  using stan::agrad::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  x++;
  EXPECT_FLOAT_EQ(0.5 + 1.0, x.val_.val_);
  EXPECT_FLOAT_EQ(1, x.val_.d_);
  EXPECT_FLOAT_EQ(0, x.d_.val_);
  EXPECT_FLOAT_EQ(0, x.d_.d_);
}
TEST(AgradFwdOperatorPlusPlus, FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  x++;
  EXPECT_FLOAT_EQ(0.5 + 1.0, x.val_.val_.val());
  EXPECT_FLOAT_EQ(1, x.val_.d_.val());
  EXPECT_FLOAT_EQ(0, x.d_.val_.val());
  EXPECT_FLOAT_EQ(0, x.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}
TEST(AgradFwdOperatorPlusPlus, FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  x++;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdOperatorPlusPlus, FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  x++;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}


struct plus_plus_fun {
  template <typename T0>
  inline T0
  operator()(T0 arg1) const {
    return (arg1++);
  }
};

TEST(AgradFwdOperatorPlusPlus, plus_plus_nan) {
  plus_plus_fun plus_plus_;

  test_nan(plus_plus_,false);
}
