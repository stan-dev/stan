#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/multiply_log.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>

TEST(AgradFwdMultiplyLog,Fvar) {
  using stan::math::fvar;
  using std::isnan;
  using std::log;
  using stan::math::multiply_log;

  fvar<double> x(0.5,1.0);
  fvar<double> y(1.2,2.0);
  fvar<double> z(-0.4,3.0);

  double w = 0.0;
  double v = 1.3;

  fvar<double> a = multiply_log(x, y);
  EXPECT_FLOAT_EQ(multiply_log(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ(1.0 * log(1.2) + 0.5 * 2.0 / 1.2, a.d_);

  fvar<double> b = multiply_log(x,z);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> c = multiply_log(x, v);
  EXPECT_FLOAT_EQ(multiply_log(0.5, 1.3), c.val_);
  EXPECT_FLOAT_EQ(log(1.3), c.d_);

  fvar<double> d = multiply_log(v, x);
  EXPECT_FLOAT_EQ(multiply_log(1.3, 0.5), d.val_);
  EXPECT_FLOAT_EQ(1.3 * 1.0 / 0.5, d.d_);

  fvar<double> e = multiply_log(x, w);
  isnan(e.val_);
  isnan(e.d_);
}

TEST(AgradFwdMultiplyLog,FvarFvarDouble) {
  using stan::math::fvar;
  using std::log;
  using stan::math::multiply_log;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.3;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.8;
  y.d_.val_ = 1.1;

  fvar<fvar<double> > a = multiply_log(x,y);

  EXPECT_FLOAT_EQ(multiply_log(1.5,1.8), a.val_.val_);
  EXPECT_FLOAT_EQ(log(1.8) * 1.3, a.val_.d_);
  EXPECT_FLOAT_EQ(1.5 / 1.8 * 1.1, a.d_.val_);
  EXPECT_FLOAT_EQ(143.0 / 180.0, a.d_.d_);
}

struct multiply_log_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return multiply_log(arg1,arg2);
  }
};

TEST(AgradFwdMultiplyLog, nan) {
  multiply_log_fun multiply_log_;
  test_nan_fwd(multiply_log_,3.0,5.0,false);
}
