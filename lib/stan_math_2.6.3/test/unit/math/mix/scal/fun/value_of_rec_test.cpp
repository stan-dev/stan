#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,value_of_rec) {
  using stan::math::var;
  using stan::math::fvar;
  using stan::math::value_of_rec;

  fvar<var> fv_a(5.0);
  fvar<fvar<var> > ffv_a(5.0);
  fvar<fvar<fvar<fvar<fvar<var> > > > > fffffv_a(5.0);

  EXPECT_FLOAT_EQ(5.0,value_of_rec(fv_a));
  EXPECT_FLOAT_EQ(5.0,value_of_rec(ffv_a));
  EXPECT_FLOAT_EQ(5.0,value_of_rec(fffffv_a));
}
