#include <stan/agrad/rev/internal/precomp_vv_vari.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(StanAgradRevInternal, precomp_vv_vari) {
  double value, gradient1, gradient2;
  AVAR x1(2), x2(3);
  AVAR y;
  
  value = 1;
  gradient1 = 4;
  gradient2 = 5;

  AVEC vars = createAVEC(x1, x2);

  EXPECT_NO_THROW(y 
      = stan::agrad::var(new stan::agrad::precomp_vv_vari(value, 
          x1.vi_, x2.vi_, gradient1, gradient2)));
  EXPECT_FLOAT_EQ(value, y.val());

  VEC g;
  EXPECT_NO_THROW(y.grad(vars, g));
  ASSERT_EQ(2U, g.size());
  EXPECT_FLOAT_EQ(gradient1, g[0]);
  EXPECT_FLOAT_EQ(gradient2, g[1]);

  stan::agrad::recover_memory();
}
