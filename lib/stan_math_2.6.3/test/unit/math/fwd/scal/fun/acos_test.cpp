#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/acos.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdAcos,Fvar) {
  using stan::math::fvar;
  using std::acos;
  using std::sqrt;
  using std::isnan;
  using stan::math::NEGATIVE_INFTY;

  fvar<double> x(0.5,1.0);
  
  fvar<double> a = acos(x);
  EXPECT_FLOAT_EQ(acos(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / -sqrt(1 - 0.5 * 0.5), a.d_);

  fvar<double> b = 2 * acos(x) + 4;
  EXPECT_FLOAT_EQ(2 * acos(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / -sqrt(1 - 0.5 * 0.5), b.d_);

  fvar<double> c = -acos(x) + 5;
  EXPECT_FLOAT_EQ(-acos(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / -sqrt(1 - 0.5 * 0.5), c.d_);

  fvar<double> d = -3 * acos(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * acos(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / -sqrt(1 - 0.5 * 0.5) + 5, d.d_);

  fvar<double> y(3.4);
  y.d_ = 1.0;
  fvar<double> e = acos(y);
  isnan(e.val_);
  isnan(e.d_);

  fvar<double> z(1.0);
  z.d_ = 1.0;
  fvar<double> f = acos(z);
  EXPECT_FLOAT_EQ(acos(1.0), f.val_);
  EXPECT_FLOAT_EQ(NEGATIVE_INFTY, f.d_);

  fvar<double> z2(1.0+stan::math::EPSILON,1.0);
  fvar<double> f2 = acos(z2);
  EXPECT_TRUE(boost::math::isnan(f2.val_));
  EXPECT_TRUE(boost::math::isnan(f2.d_));

  fvar<double> z3(-1.0-stan::math::EPSILON,1.0);
  fvar<double> f3 = acos(z3);
  EXPECT_TRUE(boost::math::isnan(f3.val_));
  EXPECT_TRUE(boost::math::isnan(f3.d_));
}

TEST(AgradFwdAcos,FvarFvarDouble) {
  using stan::math::fvar;
  using std::acos;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = acos(x);

  EXPECT_FLOAT_EQ(acos(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(-2.0 / sqrt(1.0 - 0.5 * 0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 2.0;

  a = acos(y);
  EXPECT_FLOAT_EQ(acos(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(-2.0 / sqrt(1.0 - 0.5 * 0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}


struct acos_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return acos(arg1);
  }
};

TEST(AgradFwdAcos,acos_NaN) {
  acos_fun acos_;
  test_nan_fwd(acos_,false);
}
