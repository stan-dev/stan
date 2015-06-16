#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log_sum_exp.hpp>
#include <stan/math/prim/arr/fun/log_sum_exp.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/arr/fun/log_sum_exp.hpp>
#include <stan/math/fwd/scal/fun/log_sum_exp.hpp>
#include <stan/math/fwd/scal/fun/log1p_exp.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>

TEST(AgradFwdLogSumExp,Fvar) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<double> x(0.5,1.0);
  fvar<double> y(1.2,2.0);
  double z = 1.4;

  fvar<double> a = log_sum_exp(x, y);
  EXPECT_FLOAT_EQ(log_sum_exp(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ((1.0 * exp(0.5) + 2.0 * exp(1.2)) / (exp(0.5) 
                                                       + exp(1.2)), a.d_);
  
  fvar<double> b = log_sum_exp(x, z);
  EXPECT_FLOAT_EQ(log_sum_exp(0.5, 1.4), b.val_);
  EXPECT_FLOAT_EQ(1.0 * exp(0.5) / (exp(0.5) + exp(1.4)), b.d_);
  
  fvar<double> c = log_sum_exp(z, x);
  EXPECT_FLOAT_EQ(log_sum_exp(1.4, 0.5), c.val_);
  EXPECT_FLOAT_EQ(1.0 * exp(0.5) / (exp(0.5) + exp(1.4)), c.d_);
}

TEST(AgradFwdLogSumExp,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = log_sum_exp(x,y);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), a.val_.d_);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), a.d_.val_);
  EXPECT_FLOAT_EQ(-0.045176659, a.d_.d_);
}

struct log_sum_exp_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return log_sum_exp(arg1,arg2);
  }
};

TEST(AgradFwdLogSumExp, nan) {
  log_sum_exp_fun log_sum_exp_;
  test_nan_fwd(log_sum_exp_,3.0,5.0,false);
}
