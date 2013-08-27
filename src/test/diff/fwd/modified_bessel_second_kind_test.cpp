#include <gtest/gtest.h>
#include <stan/diff/fwd.hpp>
#include <stan/diff/rev.hpp>
#include <test/diff/util.hpp>

TEST(DiffFvar, modified_bessel_second_kind) {
  using stan::diff::fvar;
  using stan::diff::modified_bessel_second_kind;

  fvar<double> a(4.0,1.0);
  int b = 1;
  fvar<double> x = modified_bessel_second_kind(b,a);
  EXPECT_FLOAT_EQ(0.0124834988872684314, x.val_);
  EXPECT_FLOAT_EQ(-0.014280550807670132, x.d_);

  fvar<double> c(-3.0,1.0);
  EXPECT_THROW(modified_bessel_second_kind(1, c), std::domain_error);
  EXPECT_THROW(modified_bessel_second_kind(-1, c), std::domain_error);
}

TEST(DiffFvarVar, modified_bessel_second_kind) {
  using stan::diff::fvar;
  using stan::diff::var;
  using stan::math::modified_bessel_second_kind;

  fvar<var> z(4.0,2.0);
  fvar<var> a = modified_bessel_second_kind(1,z);

  EXPECT_FLOAT_EQ(modified_bessel_second_kind(1, 4.0), a.val_.val());
  EXPECT_FLOAT_EQ(-0.014280550807670132 * 2.0, a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.014280550807670132, g[0]);
}

TEST(DiffFvarFvar, modified_bessel_second_kind) {
  using stan::diff::fvar;
  using stan::math::modified_bessel_second_kind;

  fvar<fvar<double> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = modified_bessel_second_kind(1,y);

  EXPECT_FLOAT_EQ(modified_bessel_second_kind(1,4.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(-0.014280550807670132, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > b = modified_bessel_second_kind(1, x);

  EXPECT_FLOAT_EQ(modified_bessel_second_kind(1,4.0), b.val_.val_);
  EXPECT_FLOAT_EQ(-0.014280550807670132, b.val_.d_);
  EXPECT_FLOAT_EQ(0, b.d_.val_);
  EXPECT_FLOAT_EQ(0, b.d_.d_);
}
