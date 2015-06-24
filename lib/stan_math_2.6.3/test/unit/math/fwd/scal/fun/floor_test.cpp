#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>

TEST(AgradFwdFloor,Fvar) {
  using stan::math::fvar;
  using std::floor;

  fvar<double> x(0.5,1.0);
  fvar<double> y(2.0,2.0);

  fvar<double> a = floor(x);
  EXPECT_FLOAT_EQ(floor(0.5), a.val_);
  EXPECT_FLOAT_EQ(0, a.d_);

  fvar<double> b = floor(y);
  EXPECT_FLOAT_EQ(floor(2.0), b.val_);
  EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c = floor(2 * x);
  EXPECT_FLOAT_EQ(floor(2 * 0.5), c.val_);
   EXPECT_FLOAT_EQ(0.0, c.d_);
}

TEST(AgradFwdFloor,FvarFvarDouble) {
  using stan::math::fvar;
  using std::floor;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = floor(x);

  EXPECT_FLOAT_EQ(floor(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = floor(y);
  EXPECT_FLOAT_EQ(floor(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct floor_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return floor(arg1);
  }
};

TEST(AgradFwdFloor,floor_NaN) {
  floor_fun floor_;
  test_nan_fwd(floor_,false);
}
