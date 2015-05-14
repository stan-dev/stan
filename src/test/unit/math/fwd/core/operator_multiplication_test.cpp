#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>

TEST(AgradFwdOperatorMultiplication, Fvar) {
  using stan::math::fvar;

  fvar<double> x1(0.5,1.0);
  fvar<double> x2(0.4,2.0);
  fvar<double> a = x1 * x2;

  EXPECT_FLOAT_EQ(0.5 * 0.4, a.val_);
  EXPECT_FLOAT_EQ(1.0 * 0.4 + 2.0 * 0.5, a.d_);

  fvar<double> b = -x1 * x2;
  EXPECT_FLOAT_EQ(-0.5 * 0.4, b.val_);
  EXPECT_FLOAT_EQ(-1 * 0.4 - 2.0 * 0.5, b.d_);

  fvar<double> c = -3 * x1 * x2;
  EXPECT_FLOAT_EQ(-3 * 0.5 * 0.4, c.val_);
  EXPECT_FLOAT_EQ(3 * (-1 * 0.4 - 2.0 * 0.5), c.d_);

  fvar<double> x3(0.5,1.0);

  fvar<double> e = 2 * x3;
  EXPECT_FLOAT_EQ(2 * 0.5, e.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, e.d_);

  fvar<double> f = x3 * -2;
  EXPECT_FLOAT_EQ(0.5 * -2, f.val_);
  EXPECT_FLOAT_EQ(1.0 * -2, f.d_);
}

TEST(AgradFwdOperatorMultiplication, FvarFvarDouble) {
  using stan::math::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > z = x * y;
  EXPECT_FLOAT_EQ(0.25, z.val_.val_);
  EXPECT_FLOAT_EQ(0.5, z.val_.d_);
  EXPECT_FLOAT_EQ(0.5, z.d_.val_);
  EXPECT_FLOAT_EQ(1, z.d_.d_);
}

struct multiply_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return arg1*arg2;
  }
};

TEST(AgradFwdOperatorMultiplication, multiply_nan) {
  multiply_fun multiply_;
  test_nan_fwd(multiply_,3.0,5.0,false);
}
