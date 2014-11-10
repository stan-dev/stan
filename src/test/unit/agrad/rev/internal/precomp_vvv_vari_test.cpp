#include <stan/agrad/rev/internal/precomp_vvv_vari.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(StanAgradRevInternal, precomp_vvv_vari) {
  double value, gradient1, gradient2, gradient3;
  stan::agrad::var x1(2), x2(3), x3(5);
  stan::agrad::var y;
  
  value = 1;
  gradient1 = 4;
  gradient2 = 5;
  gradient3 = 3;

  std::vector<stan::agrad::var> vars;
  vars.push_back(x1);
  vars.push_back(x2);
  vars.push_back(x3);

  EXPECT_NO_THROW(y 
      = stan::agrad::var(new stan::agrad::precomp_vvv_vari(value, 
          x1.vi_, x2.vi_, x3.vi_, gradient1, gradient2, gradient3)));
  EXPECT_FLOAT_EQ(value, y.val());

  std::vector<double> g;
  EXPECT_NO_THROW(y.grad(vars, g));
  ASSERT_EQ(3U, g.size());

  EXPECT_FLOAT_EQ(gradient1, g[0]);
  EXPECT_FLOAT_EQ(gradient2, g[1]);
  EXPECT_FLOAT_EQ(gradient3, g[2]);

  stan::agrad::recover_memory();
}
