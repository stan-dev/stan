#include <cmath>
#include <gtest/gtest.h>
#include <stan/math/fwd/scal/fun/fma.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>

TEST(AgradFwdFma,Fvar) { 
  using stan::math::fvar;
  fvar<double> x(0.5);
  fvar<double> y(1.2);
  fvar<double> z(1.8);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  double p = 1.4;
  double q = 2.3;

  fvar<double> a = fma(x, y, z);
  EXPECT_FLOAT_EQ(fma(0.5, 1.2, 1.8), a.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.2 + 2.0 * 0.5 + 3.0, a.d_);

  fvar<double> b = fma(p, y, z);
  EXPECT_FLOAT_EQ(fma(1.4, 1.2, 1.8), b.val_);
  EXPECT_FLOAT_EQ(2.0 * 1.4 + 3.0, b.d_);

  fvar<double> c = fma(x, p, z);
  EXPECT_FLOAT_EQ(fma(0.5, 1.4, 1.8), c.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.4 + 3.0, c.d_);

  fvar<double> d = fma(x, y, p);
  EXPECT_FLOAT_EQ(fma(0.5, 1.2, 1.4), d.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.2 + 2.0 * 0.5, d.d_);

  fvar<double> e = fma(p, q, z);
  EXPECT_FLOAT_EQ(fma(1.4, 2.3, 1.8), e.val_);
  EXPECT_FLOAT_EQ(3.0, e.d_);

  fvar<double> f = fma(x, p, q);
  EXPECT_FLOAT_EQ(fma(0.5, 1.4, 2.3), f.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.4, f.d_);

  fvar<double> g = fma(q, y, p);
  EXPECT_FLOAT_EQ(fma(2.3, 1.2, 1.4), g.val_);
  EXPECT_FLOAT_EQ(2.0 * 2.3, g.d_);
}


TEST(AgradFwdFma,FvarFvarDouble) {
  using stan::math::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > z;
  z.val_.val_ = 1.7;

  fvar<fvar<double> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_);
  EXPECT_FLOAT_EQ(1.5, a.val_.d_);
  EXPECT_FLOAT_EQ(2.5, a.d_.val_);
  EXPECT_FLOAT_EQ(1, a.d_.d_);
}

struct fma_fun {
  template <typename T0, typename T1, typename T2>
  inline
  typename stan::return_type<T0,T1,T2>::type
  operator()(const T0& arg1,
             const T1& arg2,
             const T2& arg3) const {
    return fma(arg1,arg2,arg3);
  }
};

TEST(AgradFwdFma,fma_NaN) {
  fma_fun fma_;
  test_nan_fwd(fma_,0.6,0.3,0.5,false);
}
