#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/inv_square.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/inv_square.hpp>

TEST(AgradFwdInvSquare,Fvar) {
  using stan::math::fvar;
  using stan::math::inv_square;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // Derivatives w.r.t. x
  fvar<double> a = inv_square(x);

  EXPECT_FLOAT_EQ(inv_square(0.5), a.val_);
  EXPECT_FLOAT_EQ(-2 / (0.5 * 0.5 * 0.5), a.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> g = inv_square(z);
  EXPECT_FLOAT_EQ(stan::math::positive_infinity(), g.val_);
  EXPECT_FLOAT_EQ(stan::math::negative_infinity(), g.d_);
}   

TEST(AgradFwdInvSquare,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::inv_square;
  using std::log;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = inv_square(x);

  EXPECT_FLOAT_EQ(inv_square(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(-2.0 * inv_square(0.5) / (0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct inv_square_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return inv_square(arg1);
  }
};

TEST(AgradFwdInvSquare,inv_square_NaN) {
  inv_square_fun inv_square_;
  test_nan_fwd(inv_square_,false);
}
