#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>

TEST(AgradFwdPow, Fvar) {
  using stan::math::fvar;
  using std::pow;
  using std::log;
  using std::isnan;

  fvar<double> x(0.5,1.0);
  double y = 5.0;

  fvar<double> a = pow(x, y);
  EXPECT_FLOAT_EQ(pow(0.5, 5.0), a.val_);
  EXPECT_FLOAT_EQ(5.0 * pow(0.5, 5.0 - 1.0), a.d_);

  fvar<double> b = pow(y, x);
  EXPECT_FLOAT_EQ(pow(5.0, 0.5), b.val_);
  EXPECT_FLOAT_EQ(log(5.0) * pow(5.0, 0.5), b.d_);

  fvar<double> z(1.2,2.0);
  fvar<double> c = pow(x, z);
  EXPECT_FLOAT_EQ(pow(0.5, 1.2), c.val_);
  EXPECT_FLOAT_EQ((2.0 * log(0.5) + 1.2 * 1.0 / 0.5) * pow(0.5, 1.2), c.d_);

  fvar<double> w(-0.4,1.0);
  fvar<double> d = pow(w, x);
  isnan(d.val_);
  isnan(d.d_);
}

TEST(AgradFwdPow, FvarFvarDouble) {
  using stan::math::fvar;
  using std::pow;
  using std::log;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = pow(x,y);

  EXPECT_FLOAT_EQ(pow(0.5,0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0.5 * pow(0.5,-0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(log(0.5) * pow(0.5,0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(pow(0.5, -0.5) * (0.5 * log(0.5) + 1.0), a.d_.d_);
}

struct pow_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return pow(arg1,arg2);
  }
};

TEST(AgradFwdPow, nan) {
  pow_fun pow_;
  test_nan_fwd(pow_,3.0,5.0,false);
}
