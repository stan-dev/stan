#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/logit.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/logit.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>

class AgradFwdLogit : public testing::Test {
  void SetUp() {
  }
};


TEST_F(AgradFwdLogit,Fvar) {
  using stan::math::fvar;
  using stan::math::logit;
  using std::isnan;

  fvar<double> x(0.5,1.0);

  fvar<double> a = logit(x);
  EXPECT_FLOAT_EQ(logit(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 - 0.5 * 0.5), a.d_);

  fvar<double> y(-1.2,1.0);

  fvar<double> b = logit(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> z(1.5,1.0);

  fvar<double> c = logit(z);
  isnan(c.val_);
  isnan(c.d_);
}

TEST_F(AgradFwdLogit,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::logit;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = logit(x);

  EXPECT_FLOAT_EQ(logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 - 0.25), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = logit(y);
  EXPECT_FLOAT_EQ(logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(1 / (0.5 - 0.25), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct logit_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return logit(arg1);
  }
};

TEST_F(AgradFwdLogit,logit_NaN) {
  logit_fun logit_;
  test_nan_fwd(logit_,false);
}
