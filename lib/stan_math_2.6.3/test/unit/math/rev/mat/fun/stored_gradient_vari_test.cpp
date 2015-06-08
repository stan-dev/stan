#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <gtest/gtest.h>

#include <vector>
#include <stan/math/rev/core.hpp>

TEST(StoredGradientVari, propagate3) {
  using stan::math::var;
  using stan::math::vari;
  vari** xs
    = (vari**) stan::math::ChainableStack::memalloc_.alloc(3 * sizeof(vari*));
  var xs1 = 1; // value not used here
  var xs2 = 4; // value not used here
  var xs3 = 9; // value not used here
  xs[0] = xs1.vi_;
  xs[1] = xs2.vi_;
  xs[2] = xs3.vi_;
  double* partials = (double*) stan::math::ChainableStack::memalloc_.alloc(3 * sizeof(double));
  partials[0] = 10;
  partials[1] = 100;
  partials[2] = 1000;

  var sum = var(new stan::math::stored_gradient_vari(-14.7, 3, xs, partials));
  EXPECT_FLOAT_EQ(-14.7, sum.val());

  std::vector<var> in(3);
  in[0] = var(xs1);
  in[1] = var(xs2);
  in[2] = var(xs3);

  var f = 132.7 * sum;

  std::vector<double> g;
  f.grad(in, g);
  EXPECT_EQ(3U, g.size());
  EXPECT_EQ(10 * 132.7, g[0]);
  EXPECT_EQ(100 * 132.7, g[1]);
  EXPECT_EQ(1000 * 132.7, g[2]);
}

TEST(StoredGradientVari, propagate0) {
  using stan::math::var;
  using stan::math::vari;
  vari** xs = 0;
  double* partials = (double*) 0;

  var sum = var(new stan::math::stored_gradient_vari(-14.7, 0, xs, partials));
  EXPECT_FLOAT_EQ(-14.7, sum.val());

  std::vector<var> dummy(3);
  dummy[0] = 1;
  dummy[1] = 2;
  dummy[2] = 3;

  var f = 132.7 * sum;

  std::vector<double> g;
  f.grad(dummy, g);
  EXPECT_EQ(3U, g.size());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(0, g[i]);
}
