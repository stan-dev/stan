#include <stan/agrad/rev/internal/precomp_v_vari.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(StanAgradRevInternal, precomp_v_vari) {
  double value, gradient;
  stan::agrad::var x1(2);
  stan::agrad::var y;
  
  value = 1;
  gradient = 4;

  std::vector<stan::agrad::var> wrapper;
  wrapper.push_back(x1);

  EXPECT_NO_THROW(y = stan::agrad::var(new stan::agrad::precomp_v_vari(value, x1.vi_, gradient)));
  EXPECT_FLOAT_EQ(value, y.val());

  std::vector<double> g;
  EXPECT_NO_THROW(y.grad(wrapper, g));
  ASSERT_EQ(1U, g.size());
  EXPECT_FLOAT_EQ(gradient, g[0]);

  stan::agrad::recover_memory();
}
