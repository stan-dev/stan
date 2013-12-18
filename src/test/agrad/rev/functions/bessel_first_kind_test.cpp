#include <stan/agrad/rev/functions/bessel_first_kind.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,bessel_first_kind_int_var) {
  int a(0);
  AVAR b(4.0);
  AVAR f = stan::agrad::bessel_first_kind(a,b);
  EXPECT_FLOAT_EQ(-0.39714980986384737228659076845169804197561868528938,f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0,g[0]);
  EXPECT_FLOAT_EQ(0.0660433280235491361431854208032750287274234195317, g[1]);

  a = 1;
  b = -3.0;
  f = stan::agrad::bessel_first_kind(a,b);

  EXPECT_FLOAT_EQ(-0.33905895852593645892551459720647889697308041819800,
                  f.val());

  x = createAVEC(a,b);
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0,g[0]);
  EXPECT_FLOAT_EQ(0.5 * -0.7461432154878245145319857900923154716212709191545920,g[1]);
}
