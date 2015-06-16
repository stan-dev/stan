#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/atan.hpp>
#include <stan/math/fwd/core.hpp>

class AgradFwdAtan : public testing::Test {
  void SetUp() {
  }
};


TEST_F(AgradFwdAtan,Fvar) {
  using stan::math::fvar;
  using std::atan;

  fvar<double> x(0.5,1.0);
  
  fvar<double> a = atan(x);
  EXPECT_FLOAT_EQ(atan(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 + 0.5 * 0.5), a.d_);

  fvar<double> b = 2 * atan(x) + 4;
  EXPECT_FLOAT_EQ(2 * atan(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / (1 + 0.5 * 0.5), b.d_);

  fvar<double> c = -atan(x) + 5;
  EXPECT_FLOAT_EQ(-atan(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (1 + 0.5 * 0.5), c.d_);

  fvar<double> d = -3 * atan(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * atan(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (1 + 0.5 * 0.5) + 5, d.d_);
}


TEST_F(AgradFwdAtan,FvarFvarDouble) {
  using stan::math::fvar;
  using std::atan;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = atan(x);

  EXPECT_FLOAT_EQ(atan(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 / (1.0 + 1.5 * 1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = atan(y);
  EXPECT_FLOAT_EQ(atan(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 / (1.0 + 1.5 * 1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct atan_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return atan(arg1);
  }
};

TEST_F(AgradFwdAtan,atan_NaN) {
  atan_fun atan_;
  test_nan_fwd(atan_,false);
}
