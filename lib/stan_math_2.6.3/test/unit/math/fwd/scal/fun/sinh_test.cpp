#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/sinh.hpp>
#include <stan/math/fwd/scal/fun/cosh.hpp>

class AgradFwdSinh : public testing::Test {
  void SetUp() {
  }
};



TEST_F(AgradFwdSinh, Fvar) {
  using stan::math::fvar;
  using std::sinh;
  using std::cosh;

  fvar<double> x(0.5,1.0);

  fvar<double> a = sinh(x);
  EXPECT_FLOAT_EQ(sinh(0.5), a.val_);
  EXPECT_FLOAT_EQ(cosh(0.5), a.d_);

  fvar<double> y(-1.2,1.0);

  fvar<double> b = sinh(y);
  EXPECT_FLOAT_EQ(sinh(-1.2), b.val_);
  EXPECT_FLOAT_EQ(cosh(-1.2), b.d_);

  fvar<double> c = sinh(-x);
  EXPECT_FLOAT_EQ(sinh(-0.5), c.val_);
  EXPECT_FLOAT_EQ(-cosh(-0.5), c.d_);
}


TEST_F(AgradFwdSinh, FvarFvarDouble) {
  using stan::math::fvar;
  using std::sinh;
  using std::cosh;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = sinh(x);

  EXPECT_FLOAT_EQ(sinh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 * cosh(1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = sinh(y);
  EXPECT_FLOAT_EQ(sinh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 * cosh(1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct sinh_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return sinh(arg1);
  }
};

TEST_F(AgradFwdSinh,sinh_NaN) {
  sinh_fun sinh_;
  test_nan_fwd(sinh_,false);
}
