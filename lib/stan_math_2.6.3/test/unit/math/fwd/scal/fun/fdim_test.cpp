#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/fdim.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>

TEST(AgradFwdFdim,Fvar) {
  using stan::math::fvar;
  using stan::math::fdim;
  using std::isnan;
  using std::floor;

  fvar<double> x(2.0,1.0);
  fvar<double> y(-3.0,2.0);

  fvar<double> a = fdim(x, y);
  EXPECT_FLOAT_EQ(fdim(2.0, -3.0), a.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.0 + 2.0 * -floor(2.0 / -3.0), a.d_);

  fvar<double> b = fdim(2 * x, y);
  EXPECT_FLOAT_EQ(fdim(2 * 2.0, -3.0), b.val_);
  EXPECT_FLOAT_EQ(2 * 1.0 * 1.0 + 2.0 * -floor(4.0 / -3.0), b.d_);

  fvar<double> c = fdim(y, x);
  EXPECT_FLOAT_EQ(fdim(-3.0, 2.0), c.val_);
  EXPECT_FLOAT_EQ(0.0, c.d_);

  fvar<double> d = fdim(x, x);
  EXPECT_FLOAT_EQ(fdim(2.0, 2.0), d.val_);
  EXPECT_FLOAT_EQ(0.0, d.d_);

  double z = 1.0;

  fvar<double> e = fdim(x, z);
  EXPECT_FLOAT_EQ(fdim(2.0, 1.0), e.val_);
  EXPECT_FLOAT_EQ(1.0, e.d_);

  fvar<double> f = fdim(z, x);
  EXPECT_FLOAT_EQ(fdim(1.0, 2.0), f.val_);
  EXPECT_FLOAT_EQ(0.0, f.d_);
 }


TEST(AgradFwdFdim,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::fdim;
  using std::floor;

  fvar<fvar<double> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = fdim(x,y);

  EXPECT_FLOAT_EQ(fdim(2.5,1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(1, a.val_.d_);
  EXPECT_FLOAT_EQ(-1, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct fdim_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return fdim(arg1,arg2);
  }
};

TEST(AgradFwdFdim, nan) {
  fdim_fun fdim_;
  test_nan_fwd(fdim_,3.0,5.0,false);
}
