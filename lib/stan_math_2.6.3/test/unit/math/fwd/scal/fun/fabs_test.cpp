#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>

TEST(AgradFwdFabs,Fvar) {
  using stan::math::fvar;
  using std::fabs;
  using std::isnan;

  fvar<double> x(2.0);
  fvar<double> y(-3.0);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = fabs(x);
  EXPECT_FLOAT_EQ(fabs(2), a.val_);
  EXPECT_FLOAT_EQ(1.0, a.d_);

  fvar<double> b = fabs(-x);
  EXPECT_FLOAT_EQ(fabs(-2), b.val_);
  EXPECT_FLOAT_EQ(1.0, b.d_);

  fvar<double> c = fabs(y);
  EXPECT_FLOAT_EQ(fabs(-3), c.val_);
  EXPECT_FLOAT_EQ(-2.0, c.d_);

  fvar<double> d = fabs(x * 2);
  EXPECT_FLOAT_EQ(fabs(2 * 2), d.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, d.d_);

  fvar<double> e = fabs(y + 4);
  EXPECT_FLOAT_EQ(fabs(-3.0 + 4), e.val_);
  EXPECT_FLOAT_EQ(2.0, e.d_);

  fvar<double> f = fabs(x - 2);
  EXPECT_FLOAT_EQ(fabs(2.0 - 2), f.val_);
  EXPECT_FLOAT_EQ(0, f.d_);

  fvar<double> w = std::numeric_limits<double>::quiet_NaN();
  fvar<double> h = fabs(w);
  EXPECT_TRUE(boost::math::isnan(h.val_));
  EXPECT_TRUE(boost::math::isnan(h.d_));

  fvar<double> u = 0;
  fvar<double> j = fabs(u);
  EXPECT_FLOAT_EQ(0.0, j.val_);
  EXPECT_FLOAT_EQ(0.0, j.d_);
}

TEST(AgradFwdFabs,FvarFvarDouble) {
  using stan::math::fvar;
  using std::fabs;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = fabs(x);

  EXPECT_FLOAT_EQ(fabs(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;  

  a = fabs(y);
  EXPECT_FLOAT_EQ(fabs(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}


struct fabs_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return fabs(arg1);
  }
};

TEST(AgradFwdFabs,fabs_NaN) {
  fabs_fun fabs_;
  test_nan_fwd(fabs_,false);
}
