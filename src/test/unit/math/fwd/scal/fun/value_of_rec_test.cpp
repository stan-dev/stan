#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/fwd/core.hpp>
#include <gtest/gtest.h>

TEST(AgradFwd,value_of_rec) {
  using stan::math::fvar;

  fvar<double> a = 5.0;
  fvar<fvar<double> > ff_a(5.0);
  fvar<fvar<fvar<fvar<double> > > > ffff_a(5.0);
  EXPECT_FLOAT_EQ(5.0, value_of_rec(a));
  EXPECT_FLOAT_EQ(5.0, value_of_rec(ff_a));
  EXPECT_FLOAT_EQ(5.0, value_of_rec(ffff_a));
}

