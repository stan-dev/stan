#include <stan/services/optimize/newton.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/linear_target.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <cmath>
#include <limits>

struct ServicesOptimizeNewtonLinearTarget : public testing::Test {
  ServicesOptimizeNewtonLinearTarget()
      : init(init_ss), parameter(parameter_ss), model(context, 0, &model_ss) {}

  std::stringstream init_ss, parameter_ss, model_ss;
  stan::test::unit::instrumented_logger logger;
  stan::callbacks::stream_writer init;
  stan::test::unit::values_writer parameter;
  stan::io::empty_var_context context;
  stan_model model;
};

TEST_F(ServicesOptimizeNewtonLinearTarget, does_not_report_convergence) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int num_iterations = 10;
  bool save_iterations = false;
  stan::test::unit::instrumented_interrupt interrupt;

  int return_code = stan::services::optimize::newton(
      model, context, seed, chain, init_radius, num_iterations, save_iterations,
      interrupt, logger, init, parameter);

  EXPECT_EQ(stan::services::error_codes::OK, return_code);
  ASSERT_EQ(3, parameter.names_.size());
  EXPECT_EQ("converged__", parameter.names_[1]);
  EXPECT_EQ("x", parameter.names_[2]);
  ASSERT_EQ(1, parameter.states_.size());

  double converged = parameter.states_.back()[1];
  double x = parameter.states_.back()[2];
  EXPECT_EQ(stan::optimization::TERM_MAXIT, converged)
      << "a linear target has no mode, so it must not report convergence";
  EXPECT_TRUE(std::isfinite(x)) << "final x = " << x;
  EXPECT_GT(x, 0.0) << "optimizer must have moved uphill";
  const double sqrt_eps = std::sqrt(std::numeric_limits<double>::epsilon());
  EXPECT_LE(x, num_iterations * 2.0 / sqrt_eps)
      << "each step along a zero-curvature direction must be bounded";
}
