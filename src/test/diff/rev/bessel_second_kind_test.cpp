#include <stan/diff/rev/bessel_second_kind.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,bessel_second_kind_int_var) {
  int a(0);
  AVAR b(4.0);
  AVAR f = stan::diff::bessel_second_kind(a,b);
  EXPECT_FLOAT_EQ(-0.01694073932506499190363513444715321824049258989801,f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0,g[0]);
  EXPECT_FLOAT_EQ(-0.39792571055710000525397997245079185227118918162290, g[1]);

  a = 1;
  b = 3.0;
  f = stan::diff::bessel_second_kind(a,b);

  EXPECT_FLOAT_EQ(0.3246744247917999784370128392879532396692751433723549,
                  f.val());

  x = createAVEC(a,b);
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0,g[0]);
  EXPECT_FLOAT_EQ(0.5 * 0.5372504034977141116428784919345973293208741759303264,
                  g[1]);

  b = -4.0;
  EXPECT_THROW(stan::diff::bessel_second_kind(0,b), std::domain_error);
}
