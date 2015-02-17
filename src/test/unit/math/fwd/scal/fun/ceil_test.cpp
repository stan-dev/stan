#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/operator_addition.hpp>
#include <stan/math/fwd/scal/fun/operator_division.hpp>
#include <stan/math/fwd/scal/fun/operator_equal.hpp>
#include <stan/math/fwd/scal/fun/operator_greater_than.hpp>
#include <stan/math/fwd/scal/fun/operator_greater_than_or_equal.hpp>
#include <stan/math/fwd/scal/fun/operator_less_than.hpp>
#include <stan/math/fwd/scal/fun/operator_less_than_or_equal.hpp>
#include <stan/math/fwd/scal/fun/operator_multiplication.hpp>
#include <stan/math/fwd/scal/fun/operator_not_equal.hpp>
#include <stan/math/fwd/scal/fun/operator_subtraction.hpp>
#include <stan/math/fwd/scal/fun/operator_unary_minus.hpp>
#include <stan/math/rev/scal/fun/operator_addition.hpp>
#include <stan/math/rev/scal/fun/operator_divide_equal.hpp>
#include <stan/math/rev/scal/fun/operator_division.hpp>
#include <stan/math/rev/scal/fun/operator_equal.hpp>
#include <stan/math/rev/scal/fun/operator_greater_than.hpp>
#include <stan/math/rev/scal/fun/operator_greater_than_or_equal.hpp>
#include <stan/math/rev/scal/fun/operator_less_than.hpp>
#include <stan/math/rev/scal/fun/operator_less_than_or_equal.hpp>
#include <stan/math/rev/scal/fun/operator_minus_equal.hpp>
#include <stan/math/rev/scal/fun/operator_multiplication.hpp>
#include <stan/math/rev/scal/fun/operator_multiply_equal.hpp>
#include <stan/math/rev/scal/fun/operator_not_equal.hpp>
#include <stan/math/rev/scal/fun/operator_plus_equal.hpp>
#include <stan/math/rev/scal/fun/operator_subtraction.hpp>
#include <stan/math/rev/scal/fun/operator_unary_decrement.hpp>
#include <stan/math/rev/scal/fun/operator_unary_increment.hpp>
#include <stan/math/rev/scal/fun/operator_unary_negative.hpp>
#include <stan/math/rev/scal/fun/operator_unary_not.hpp>
#include <stan/math/rev/scal/fun/operator_unary_plus.hpp>
#include <stan/math/fwd/scal/fun/ceil.hpp>
#include <stan/math/rev/scal/fun/ceil.hpp>
#include <stan/math/fwd/core/numeric_limits.hpp>
#include <stan/math/rev/scal/meta/numeric_limits.hpp>

TEST(AgradFwdCeil,Fvar) {
  using stan::agrad::fvar;
  using std::ceil;

  fvar<double> x(0.5,1.0);
  fvar<double> y(2.0,2.0);

  fvar<double> a = ceil(x);
  EXPECT_FLOAT_EQ(ceil(0.5), a.val_);
  EXPECT_FLOAT_EQ(0, a.d_);

  fvar<double> b = ceil(y);
  EXPECT_FLOAT_EQ(ceil(2.0), b.val_);
   EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c = ceil(2 * x);
  EXPECT_FLOAT_EQ(ceil(2 * 0.5), c.val_);
   EXPECT_FLOAT_EQ(0.0, c.d_);
}

TEST(AgradFwdCeil,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::ceil;

  fvar<var> x(1.5,1.3);
  fvar<var> a = ceil(x);

  EXPECT_FLOAT_EQ(ceil(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFwdCeil,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::ceil;

  fvar<var> x(1.5,1.3);
  fvar<var> a = ceil(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFwdCeil,FvarFvarDouble) {
  using stan::agrad::fvar;
  using std::ceil;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = ceil(x);

  EXPECT_FLOAT_EQ(ceil(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = ceil(y);
  EXPECT_FLOAT_EQ(ceil(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

TEST(AgradFwdCeil,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::ceil;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = ceil(x);

  EXPECT_FLOAT_EQ(ceil(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = ceil(y);
  EXPECT_FLOAT_EQ(ceil(1.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(0, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0.0, r[0]);
}

TEST(AgradFwdCeil,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::ceil;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = ceil(x);

  EXPECT_FLOAT_EQ(ceil(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = ceil(y);
  EXPECT_FLOAT_EQ(ceil(1.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(0, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0.0, r[0]);
}
TEST(AgradFwdCeil,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::ceil;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;
  x.d_.val_ = 2.0;

  fvar<fvar<var> > a = ceil(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

struct ceil_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return ceil(arg1);
  }
};

TEST(AgradFwdCeil,ceil_NaN) {
  ceil_fun ceil_;
  test_nan(ceil_,false);
}
