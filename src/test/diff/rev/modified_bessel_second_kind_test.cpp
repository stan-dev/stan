#include <stan/diff/rev/modified_bessel_second_kind.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,modified_bessel_second_kind_int_var) {
  int a(1);
  AVAR b(4.0);
  AVAR f = stan::diff::modified_bessel_second_kind(a,b);
  EXPECT_FLOAT_EQ(0.01248349888726843147038417998080606848384,f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0,g[0]);
  EXPECT_FLOAT_EQ(-0.01428055080767013213734124, g[1]);

  a = 1;
  b = -3.0;
  EXPECT_THROW(stan::diff::modified_bessel_second_kind(a,b), std::domain_error);

  a = -1;
  EXPECT_THROW(stan::diff::modified_bessel_second_kind(a,b), std::domain_error);
}
