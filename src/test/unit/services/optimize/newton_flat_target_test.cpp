#include <stan/services/optimize/newton.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/flat_target.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <cmath>

struct ServicesOptimizeNewtonFlatTarget : public testing::Test {
  ServicesOptimizeNewtonFlatTarget()
      : init(init_ss), parameter(parameter_ss), model(context, 0, &model_ss) {}

  std::stringstream init_ss, parameter_ss, model_ss;
  stan::test::unit::instrumented_logger logger;
  stan::callbacks::stream_writer init;
  stan::test::unit::values_writer parameter;
  stan::io::empty_var_context context;
  stan_model model;
};

// Regression test for https://github.com/stan-dev/stan/issues/3425
// The service must not report success while writing non-finite parameters.
TEST_F(ServicesOptimizeNewtonFlatTarget, does_not_report_ok_with_nan_params) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 1;
  int num_iterations = 10;
  bool save_iterations = false;
  stan::test::unit::instrumented_interrupt interrupt;

  int return_code = stan::services::optimize::newton(
      model, context, seed, chain, init_radius, num_iterations, save_iterations,
      interrupt, logger, init, parameter);

  ASSERT_EQ(3, parameter.names_.size());
  EXPECT_EQ("x", parameter.names_[2]);
  ASSERT_EQ(1, parameter.states_.size());

  double x = parameter.states_.back()[2];
  EXPECT_TRUE(std::isfinite(x)
              || return_code != stan::services::error_codes::OK)
      << "newton returned error_codes::OK with x = " << x;
  EXPECT_TRUE(std::isfinite(x)) << "final x = " << x;
}
