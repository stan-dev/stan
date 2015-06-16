#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>

TEST(StanAgradRevInternal, precomp_v_vari) {
  double value, gradient;
  AVAR x1(2);
  AVAR y;
  
  value = 1;
  gradient = 4;

  AVEC wrapper = createAVEC(x1);

  EXPECT_NO_THROW(y = stan::math::var(new stan::math::precomp_v_vari(value, x1.vi_, gradient)));
  EXPECT_FLOAT_EQ(value, y.val());

  VEC g;
  EXPECT_NO_THROW(y.grad(wrapper, g));
  ASSERT_EQ(1U, g.size());
  EXPECT_FLOAT_EQ(gradient, g[0]);

  stan::math::recover_memory();
}
