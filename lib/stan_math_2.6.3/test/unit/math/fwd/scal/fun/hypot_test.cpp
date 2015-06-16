#include <gtest/gtest.h>
#include <stan/math/fwd/scal/fun/inv.hpp>
#include <stan/math/prim/scal/fun/inv.hpp>
#include <stan/math/fwd/scal/fun/hypot.hpp>
#include <boost/math/special_functions/hypot.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>

TEST(AgradFwdHypot,Fvar) {
  using stan::math::fvar;
  using boost::math::hypot;
  using std::isnan;

  fvar<double> x(0.5,1.0);
  fvar<double> y(2.3,2.0);

  fvar<double> a = hypot(x, y);
  EXPECT_FLOAT_EQ(hypot(0.5, 2.3), a.val_);
  EXPECT_FLOAT_EQ((0.5 * 1.0 + 2.3 * 2.0) / hypot(0.5, 2.3), a.d_);

  fvar<double> z(0.0,1.0);
  fvar<double> w(-2.3,2.0);
  fvar<double> b = hypot(x, z);

  EXPECT_FLOAT_EQ(0.5, b.val_);
  EXPECT_FLOAT_EQ(1.0, b.d_);

  fvar<double> c = hypot(x, w);
  isnan(c.val_);
  isnan(c.d_);

  fvar<double> d = hypot(z, x);
  EXPECT_FLOAT_EQ(0.5, d.val_);
  EXPECT_FLOAT_EQ(1.0, d.d_);
}

TEST(AgradFwdHypot,FvarFvarDouble) {
  using stan::math::fvar;
  using boost::math::hypot;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = hypot(x,y);

  EXPECT_FLOAT_EQ(hypot(3.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(3.0 / hypot(3.0,6.0), a.val_.d_);
  EXPECT_FLOAT_EQ(6.0 / hypot(3.0,6.0), a.d_.val_);
  EXPECT_FLOAT_EQ(-0.059628479, a.d_.d_);
}

struct hypot_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return hypot(arg1,arg2);
  }
};

TEST(AgradFwdHypot, nan) {
  hypot_fun hypot_;
  test_nan_fwd(hypot_,3.0,5.0,false);
}
