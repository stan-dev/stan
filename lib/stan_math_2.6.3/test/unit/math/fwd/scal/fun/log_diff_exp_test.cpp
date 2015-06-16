#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log_diff_exp.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/log_diff_exp.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>

TEST(AgradFwdLogDiffExp,Fvar) {
  using stan::math::fvar;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<double> x(1.2);
  fvar<double> y(0.5);
  x.d_ = 1.0;
  y.d_ = 2.0;

  double z = 1.1;

  fvar<double> a = log_diff_exp(x, y);
  EXPECT_FLOAT_EQ(log_diff_exp(1.2, 0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 - exp(0.5 - 1.2) ) + 2 / (1 - exp(1.2 - 0.5) ), a.d_);

  fvar<double> b = log_diff_exp(x, z);
  EXPECT_FLOAT_EQ(log_diff_exp(1.2, 1.1), b.val_);
  EXPECT_FLOAT_EQ(1 / (1 - exp(1.1 - 1.2) ), b.d_);

  fvar<double> c = log_diff_exp(z, y);
  EXPECT_FLOAT_EQ(log_diff_exp(1.1, 0.5), c.val_);
  EXPECT_FLOAT_EQ(2 / (1 - exp(1.1 - 0.5) ), c.d_);
}

TEST(AgradFwdLogDiffExp, AgradFvar_exception) {
  using stan::math::fvar;
  EXPECT_NO_THROW(log_diff_exp(fvar<double>(3), fvar<double>(4)));
  EXPECT_NO_THROW(log_diff_exp(fvar<double>(3), 4));
  EXPECT_NO_THROW(log_diff_exp(3, fvar<double>(4)));
}

TEST(AgradFwdLogDiffExp,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = 9.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = log_diff_exp(x,y);

  EXPECT_FLOAT_EQ(log_diff_exp(9.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(9.0) / (exp(9.0) - exp(6.0)), a.val_.d_);
  EXPECT_FLOAT_EQ(-exp(6.0) / (exp(9.0) - exp(6.0)), a.d_.val_);
  EXPECT_FLOAT_EQ(0.055141006, a.d_.d_);
}

struct log_diff_exp_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return log_diff_exp(arg1,arg2);
  }
};

TEST(AgradFwdLogDiffExp, nan) {
  log_diff_exp_fun log_diff_exp_;
  test_nan_fwd(log_diff_exp_,3.0,5.0,false);
}
