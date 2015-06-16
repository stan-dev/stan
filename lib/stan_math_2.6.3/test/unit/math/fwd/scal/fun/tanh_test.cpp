#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/tanh.hpp>

class AgradFwdTanh : public testing::Test {
  void SetUp() {
  }
};


TEST_F(AgradFwdTanh, Fvar) {
  using stan::math::fvar;
  using std::tanh;

  fvar<double> x(0.5,1.0);
  fvar<double> a = tanh(x);
  EXPECT_FLOAT_EQ(tanh(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 - tanh(0.5) * tanh(0.5), a.d_);

  fvar<double> y(-1.2,1.0);
  fvar<double> b = tanh(y);
  EXPECT_FLOAT_EQ(tanh(-1.2), b.val_);
  EXPECT_FLOAT_EQ(1 - tanh(-1.2) * tanh(-1.2), b.d_);

  fvar<double> c = tanh(-x);
  EXPECT_FLOAT_EQ(tanh(-0.5), c.val_);
  EXPECT_FLOAT_EQ(-1 * (1 - tanh(-0.5) * tanh(-0.5)), c.d_);
}

TEST_F(AgradFwdTanh, FvarFvarDouble) {
  using stan::math::fvar;
  using std::tanh;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = tanh(x);

  EXPECT_FLOAT_EQ(tanh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 * (1.0 - tanh(1.5) * tanh(1.5)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = tanh(y);
  EXPECT_FLOAT_EQ(tanh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 * (1.0 - tanh(1.5) * tanh(1.5)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct tanh_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return tanh(arg1);
  }
};

TEST_F(AgradFwdTanh,tanh_NaN) {
  tanh_fun tanh_;
  test_nan_fwd(tanh_,false);
}
