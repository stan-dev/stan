#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/fmin.hpp>

TEST(AgradFwdFmin,Fvar) {
  using stan::math::fvar;
  using stan::math::fmin;
  using std::isnan;

  fvar<double> x(2.0,1.0);
  fvar<double> y(-3.0,2.0);

  fvar<double> a = fmin(x, y);
  EXPECT_FLOAT_EQ(-3.0, a.val_);
  EXPECT_FLOAT_EQ(2.0, a.d_);

  fvar<double> b = fmin(2 * x, y);
  EXPECT_FLOAT_EQ(-3.0, b.val_);
  EXPECT_FLOAT_EQ(2.0, b.d_);

  fvar<double> c = fmin(y, x);
  EXPECT_FLOAT_EQ(-3.0, c.val_);
  EXPECT_FLOAT_EQ(2.0, c.d_);

  fvar<double> d = fmin(x, x);
  EXPECT_FLOAT_EQ(2.0, d.val_);
  isnan(d.d_);

  double z = 1.0;

  fvar<double> e = fmin(x, z);
  EXPECT_FLOAT_EQ(1.0, e.val_);
  EXPECT_FLOAT_EQ(0.0, e.d_);

  fvar<double> f = fmin(z, x);
  EXPECT_FLOAT_EQ(1.0, f.val_);
  EXPECT_FLOAT_EQ(0.0, f.d_);
 }

TEST(AgradFwdFmin,FvarFvarDouble) {
  using stan::math::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = fmin(x,y);

  EXPECT_FLOAT_EQ(fmin(2.5,1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(1, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct fmin_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return fmin(arg1,arg2);
  }
};

TEST(AgradFwdFmin, nan) {
  fmin_fun fmin_;
  double nan = std::numeric_limits<double>::quiet_NaN();
  test_nan_fwd(fmin_, nan, nan, false);
}
