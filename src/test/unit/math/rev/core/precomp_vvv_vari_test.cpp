#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>

TEST(StanAgradRevInternal, precomp_vvv_vari) {
  double value, gradient1, gradient2, gradient3;
  AVAR x1(2), x2(3), x3(5);
  AVAR y;
  
  value = 1;
  gradient1 = 4;
  gradient2 = 5;
  gradient3 = 3;

  AVEC vars = createAVEC(x1, x2, x3);

  EXPECT_NO_THROW(y 
      = stan::math::var(new stan::math::precomp_vvv_vari(value, 
          x1.vi_, x2.vi_, x3.vi_, gradient1, gradient2, gradient3)));
  EXPECT_FLOAT_EQ(value, y.val());

  VEC g;
  EXPECT_NO_THROW(y.grad(vars, g));
  ASSERT_EQ(3U, g.size());

  EXPECT_FLOAT_EQ(gradient1, g[0]);
  EXPECT_FLOAT_EQ(gradient2, g[1]);
  EXPECT_FLOAT_EQ(gradient3, g[2]);

  stan::math::recover_memory();
}
