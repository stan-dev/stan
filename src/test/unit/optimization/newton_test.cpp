#include <gtest/gtest.h>
#include <stan/optimization/newton.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/flat_target.hpp>
#include <cmath>
#include <vector>

typedef flat_target_model_namespace::flat_target_model Model;

// Regression test for https://github.com/stan-dev/stan/issues/3425
TEST(OptimizationNewton, flat_direction_keeps_parameters_finite) {
  stan::io::empty_var_context dummy_context;
  Model model(dummy_context);

  std::vector<double> params_r(1, 1.0);
  std::vector<int> params_i;

  double f = stan::optimization::newton_step<Model, false>(model, params_r,
                                                           params_i);

  EXPECT_FLOAT_EQ(0.5, f);
  ASSERT_EQ(1u, params_r.size());
  EXPECT_TRUE(std::isfinite(params_r[0]))
      << "newton_step produced non-finite parameter: " << params_r[0];
}

TEST(OptimizationNewton, make_negative_definite_and_solve_zero_hessian) {
  stan::optimization::matrix_d H = stan::optimization::matrix_d::Zero(2, 2);
  stan::optimization::vector_d g = stan::optimization::vector_d::Zero(2);

  stan::optimization::make_negative_definite_and_solve(H, g);

  for (int i = 0; i < g.size(); ++i) {
    EXPECT_TRUE(std::isfinite(g[i]))
        << "step direction has non-finite component " << i << ": " << g[i];
  }
}
