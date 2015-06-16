#include <gtest/gtest.h>
#include <boost/math/special_functions/asinh.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/asinh.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/core.hpp>

class AgradFwdAsinh : public testing::Test {
  void SetUp() {
  }
};

TEST_F(AgradFwdAsinh,Fvar) {
  using stan::math::fvar;
  using boost::math::asinh;
  using std::sqrt;

  fvar<double> x(0.5,1.0);

  fvar<double> a = asinh(x);
  EXPECT_FLOAT_EQ(asinh(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(1 + (0.5) * (0.5)), a.d_);

  fvar<double> y(-1.2,1.0);

  fvar<double> b = asinh(y);
  EXPECT_FLOAT_EQ(asinh(-1.2), b.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(1 + (-1.2) * (-1.2)), b.d_);

  fvar<double> c = asinh(-x);
  EXPECT_FLOAT_EQ(asinh(-0.5), c.val_);
  EXPECT_FLOAT_EQ(-1 / sqrt(1 + (-0.5) * (-0.5)), c.d_);
}


TEST_F(AgradFwdAsinh,FvarFvarDouble) {
  using stan::math::fvar;
  using boost::math::asinh;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = asinh(x);

  EXPECT_FLOAT_EQ(asinh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 + 1.5 * 1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = asinh(y);
  EXPECT_FLOAT_EQ(asinh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 + 1.5 * 1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
struct asinh_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return asinh(arg1);
  }
};

TEST_F(AgradFwdAsinh,asinh_NaN) {
  asinh_fun asinh_;
  test_nan_fwd(asinh_,false);
}
