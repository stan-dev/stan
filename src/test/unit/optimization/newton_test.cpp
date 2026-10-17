#include <gtest/gtest.h>
#include <stan/optimization/newton.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/flat_target.hpp>
#include <cmath>
#include <limits>
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

TEST(OptimizationNewton,
     make_negative_definite_and_solve_floors_small_eigenvalue_at_sqrt_eps) {
  const double eps = std::numeric_limits<double>::epsilon();
  const double sqrt_eps = std::sqrt(eps);
  stan::optimization::matrix_d H = stan::optimization::matrix_d::Zero(2, 2);
  H(0, 0) = -1.0;
  H(1, 1) = -4.0 * eps;
  stan::optimization::vector_d g = stan::optimization::vector_d::Ones(2);

  stan::optimization::make_negative_definite_and_solve(H, g);

  EXPECT_FLOAT_EQ(-1.0, g[0]);
  EXPECT_FLOAT_EQ(-1.0 / sqrt_eps, g[1])
      << "eigenvalue below sqrt(eps) * max should be floored, not dropped";
}

TEST(OptimizationNewton,
     make_negative_definite_and_solve_zero_eigenvalue_uses_relative_floor) {
  const double sqrt_eps = std::sqrt(std::numeric_limits<double>::epsilon());
  stan::optimization::matrix_d H = stan::optimization::matrix_d::Zero(2, 2);
  H(0, 0) = -1.0;
  stan::optimization::vector_d g = stan::optimization::vector_d::Ones(2);

  stan::optimization::make_negative_definite_and_solve(H, g);

  EXPECT_FLOAT_EQ(-1.0, g[0]);
  EXPECT_FLOAT_EQ(-1.0 / sqrt_eps, g[1]);
}

TEST(OptimizationNewton,
     make_negative_definite_and_solve_is_continuous_at_old_cutoff) {
  const double eps = std::numeric_limits<double>::epsilon();
  const double old_cutoff = 4.0 * 2 * eps;
  stan::optimization::matrix_d H_above
      = stan::optimization::matrix_d::Zero(2, 2);
  H_above(0, 0) = -1.0;
  H_above(1, 1) = -1.01 * old_cutoff;
  stan::optimization::matrix_d H_below = H_above;
  H_below(1, 1) = -0.99 * old_cutoff;
  stan::optimization::vector_d g_above = stan::optimization::vector_d::Ones(2);
  stan::optimization::vector_d g_below = g_above;

  stan::optimization::make_negative_definite_and_solve(H_above, g_above);
  stan::optimization::make_negative_definite_and_solve(H_below, g_below);

  EXPECT_NEAR(g_above[1], g_below[1], 1e-6 * std::fabs(g_above[1]))
      << "step must not jump when an eigenvalue crosses the cutoff";
}

TEST(OptimizationNewton,
     make_negative_definite_and_solve_zero_hessian_nonzero_gradient) {
  const double sqrt_eps = std::sqrt(std::numeric_limits<double>::epsilon());
  stan::optimization::matrix_d H = stan::optimization::matrix_d::Zero(2, 2);
  stan::optimization::vector_d g = stan::optimization::vector_d::Ones(2);

  stan::optimization::make_negative_definite_and_solve(H, g);

  for (int i = 0; i < g.size(); ++i) {
    EXPECT_FLOAT_EQ(-1.0 / sqrt_eps, g[i])
        << "all-zero Hessian must use the absolute floor, component " << i;
  }
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
