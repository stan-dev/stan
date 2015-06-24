#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdOperatorEqual, Fvar) {
  using stan::math::fvar;
  fvar<double> v4 = 4;
  fvar<double> v5 = 5;
  double d4 = 4;
  double d5 = 5;
  
  EXPECT_FALSE(v5 == v4);
  EXPECT_FALSE(d5 == v4);
  EXPECT_FALSE(v5 == d4);
  EXPECT_FALSE(d5 == d4);
  EXPECT_TRUE(d5 == v5);

  int i4 = 4;
  int i5 = 5;
  int i6 = 5;

  EXPECT_FALSE(i5 == v4);
  EXPECT_FALSE(v5 == i4);
  EXPECT_FALSE(i5 == i4);
  EXPECT_FALSE(d5 == i4);
  EXPECT_FALSE(i5 == d4);
  EXPECT_TRUE(i6 == i5);
  EXPECT_TRUE(i6 == v5);
}

TEST(AgradFwdOperatorEqual, FvarFvarDouble) {
  using stan::math::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > z;
  z.val_.val_ = 0.5;
  z.d_.val_ = 1.0;

  EXPECT_FALSE(x == y);
  EXPECT_FALSE(x == z);
  EXPECT_TRUE(z == y);
}

TEST(AgradFwdOperatorEqual, eq_nan) {
  using stan::math::fvar;
  double nan = std::numeric_limits<double>::quiet_NaN();
  double a = 3.0;
  fvar<double> nan_fd = std::numeric_limits<double>::quiet_NaN();
  fvar<double> a_fd = 3.0;
  fvar<fvar<double> > nan_ffd = std::numeric_limits<double>::quiet_NaN();
  fvar<fvar<double> > a_ffd = 3.0;

  EXPECT_FALSE(a == nan_fd);
  EXPECT_FALSE(a_fd == nan_fd);
  EXPECT_FALSE(nan == nan_fd);
  EXPECT_FALSE(nan_fd == nan_fd);
  EXPECT_FALSE(a_fd == nan);
  EXPECT_FALSE(nan_fd == nan);
  EXPECT_FALSE(nan_fd == a);
  EXPECT_FALSE(nan_fd == a_fd);
  EXPECT_FALSE(nan == a_fd);

  EXPECT_FALSE(a == nan_ffd);
  EXPECT_FALSE(a_ffd == nan_ffd);
  EXPECT_FALSE(nan == nan_ffd);
  EXPECT_FALSE(nan_ffd == nan_ffd);
  EXPECT_FALSE(a_ffd == nan);
  EXPECT_FALSE(nan_ffd == nan);
  EXPECT_FALSE(nan_ffd == a);
  EXPECT_FALSE(nan_ffd == a_ffd);
  EXPECT_FALSE(nan == a_ffd);
}
