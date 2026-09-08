#include <gtest/gtest.h>
#include <stan/optimization/newton.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/linear_target.hpp>
#include <cmath>
#include <limits>
#include <vector>

typedef linear_target_model_namespace::linear_target_model Model;

TEST(OptimizationNewton, linear_target_moves_uphill_by_bounded_step) {
  const double sqrt_eps = std::sqrt(std::numeric_limits<double>::epsilon());
  stan::io::empty_var_context dummy_context;
  Model model(dummy_context);

  std::vector<double> params_r(1, 0.0);
  std::vector<int> params_i;

  double f = stan::optimization::newton_step<Model, false>(model, params_r,
                                                           params_i);

  ASSERT_EQ(1u, params_r.size());
  EXPECT_TRUE(std::isfinite(params_r[0]));
  EXPECT_GT(params_r[0], 0.0) << "zero-curvature direction must still move";
  EXPECT_LE(params_r[0], 2.0 / sqrt_eps)
      << "step along a zero-curvature direction must be bounded by the floor";
  EXPECT_GT(f, 0.0) << "objective must improve along a linear target";
}
