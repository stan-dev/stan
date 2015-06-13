#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdOperatorNotEqual,Fvar) {
  using stan::math::fvar;
  fvar<double> v4 = 4;
  fvar<double> v5 = 5;
  double d4 = 4;
  double d5 = 5;
  
  EXPECT_TRUE(v5 != v4);
  EXPECT_TRUE(d5 != v4);
  EXPECT_TRUE(v5 != d4);
  EXPECT_TRUE(d5 != d4);

  int i4 = 4;
  int i5 = 5;
  int i6 = 5;

  EXPECT_TRUE(i5 != v4);
  EXPECT_TRUE(v5 != i4);
  EXPECT_TRUE(i5 != i4);
  EXPECT_TRUE(d5 != i4);
  EXPECT_TRUE(i5 != d4);
  EXPECT_FALSE(i6 != i5);
  EXPECT_FALSE(i6 != v5);
}

TEST(AgradFwdOperatorNotEqual, FvarFvarDouble) {
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

  EXPECT_TRUE(x != y);
  EXPECT_TRUE(x != z);
  EXPECT_FALSE(z != y);
}

TEST(AgradFwdOperatorNotEqual, ne_nan) {
  using stan::math::fvar;
  double nan = std::numeric_limits<double>::quiet_NaN();
  double a = 3.0;
  fvar<double> nan_fd = std::numeric_limits<double>::quiet_NaN();
  fvar<double> a_fd = 3.0;
  fvar<fvar<double> > nan_ffd = std::numeric_limits<double>::quiet_NaN();
  fvar<fvar<double> > a_ffd = 3.0;

  EXPECT_TRUE(a != nan_fd);
  EXPECT_TRUE(a_fd != nan_fd);
  EXPECT_TRUE(nan != nan_fd);
  EXPECT_TRUE(nan_fd != nan_fd);
  EXPECT_TRUE(a_fd != nan);
  EXPECT_TRUE(nan_fd != nan);
  EXPECT_TRUE(nan_fd != a);
  EXPECT_TRUE(nan_fd != a_fd);
  EXPECT_TRUE(nan != a_fd);

  EXPECT_TRUE(a != nan_ffd);
  EXPECT_TRUE(a_ffd != nan_ffd);
  EXPECT_TRUE(nan != nan_ffd);
  EXPECT_TRUE(nan_ffd != nan_ffd);
  EXPECT_TRUE(a_ffd != nan);
  EXPECT_TRUE(nan_ffd != nan);
  EXPECT_TRUE(nan_ffd != a);
  EXPECT_TRUE(nan_ffd != a_ffd);
  EXPECT_TRUE(nan != a_ffd);
}
