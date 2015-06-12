#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/atan2.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdAtan2,Fvar) {
  using stan::math::fvar;
  using std::atan2;

  fvar<double> x(0.5,1.0);
  fvar<double> y(2.3,2.0);
  double w = 2.1;

  fvar<double> a = atan2(x, y);
  EXPECT_FLOAT_EQ(atan2(0.5, 2.3), a.val_);
  EXPECT_FLOAT_EQ((1.0 * 2.3 - 0.5 * 2.0) / (0.5 * 0.5 + 2.3 * 2.3), a.d_);

  fvar<double> b = atan2(w, x);
  EXPECT_FLOAT_EQ(atan2(2.1, 0.5), b.val_);
  EXPECT_FLOAT_EQ((-2.1 * 1.0) / (2.1 * 2.1 + 0.5 * 0.5), b.d_);

  fvar<double> c = atan2(x, w);
  EXPECT_FLOAT_EQ(atan2(0.5, 2.1), c.val_);
  EXPECT_FLOAT_EQ((1.0 * 2.1) / (0.5 * 0.5 + 2.1 * 2.1), c.d_);
}


TEST(AgradFwdAtan2,FvarFvarDouble) {
  using stan::math::fvar;
  using std::atan2;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  double z = 1.5;

  fvar<fvar<double> > a = atan2(x,y);

  EXPECT_FLOAT_EQ(atan(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(-1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.d_.val_);
  EXPECT_FLOAT_EQ((1.5 * 1.5 - 1.5 * 1.5) / ((1.5 * 1.5 + 1.5 * 1.5) * (1.5 * 1.5 + 1.5 * 1.5)), a.d_.d_);

  a = atan2(x,z);
  EXPECT_FLOAT_EQ(atan(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0.0, a.d_.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_.d_);

  a = atan2(z, y);

  EXPECT_FLOAT_EQ(atan(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0.0, a.val_.d_);
  EXPECT_FLOAT_EQ(-1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_.d_);
}


struct atan2_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return atan2(arg1,arg2);
  }
};

TEST(AgradFwdAtan2, nan) {
  atan2_fun atan2_;
  test_nan_fwd(atan2_,3.0,5.0,false);
}
