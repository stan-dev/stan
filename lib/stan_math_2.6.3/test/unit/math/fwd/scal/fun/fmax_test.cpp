#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/fmax.hpp>

TEST(AgradFwdFmax,Fvar) {
  using stan::math::fvar;
  using std::isnan;

  fvar<double> x(2.0,1.0);
  fvar<double> y(-3.0,2.0);

  fvar<double> a = fmax(x, y);
  EXPECT_FLOAT_EQ(2.0, a.val_);
  EXPECT_FLOAT_EQ(1.0, a.d_);

  fvar<double> b = fmax(2 * x, y);
  EXPECT_FLOAT_EQ(4.0, b.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, b.d_);

  fvar<double> c = fmax(y, x);
  EXPECT_FLOAT_EQ(2.0, c.val_);
  EXPECT_FLOAT_EQ(1.0, c.d_);

  fvar<double> d = fmax(x, x);
  EXPECT_FLOAT_EQ(2.0, d.val_);
  isnan(d.d_);

  double z = 1.0;

  fvar<double> e = fmax(x, z);
  EXPECT_FLOAT_EQ(2.0, e.val_);
  EXPECT_FLOAT_EQ(1.0, e.d_);

  fvar<double> f = fmax(z, x);
  EXPECT_FLOAT_EQ(2.0, f.val_);
  EXPECT_FLOAT_EQ(1.0, f.d_);
 }

TEST(AgradFwdFmax,FvarFvarDouble) {
  using stan::math::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = fmax(x,y);

  EXPECT_FLOAT_EQ(fmax(2.5,1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(1, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct fmax_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return fmax(arg1,arg2);
  }
};

TEST(AgradFwdFmax, nan) {
  fmax_fun fmax_;
  double nan = std::numeric_limits<double>::quiet_NaN();
  test_nan_fwd(fmax_, nan, nan, false);
}
