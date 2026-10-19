#include <stan/services/optimize/bfgs.hpp>
#include <stan/services/optimize/lbfgs.hpp>
#include <gtest/gtest.h>
#include <stan/io/json/json_data.hpp>
#include <test/test-models/good/optimization/two_scale.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/io/empty_var_context.hpp>
#include <sstream>

// At (x1, x2) = (3e-9, 0) the gradient in x1 is 3000 and the gradient in x2
// is 5, so the capped first step moves x2 by about 2e-12. The change of x
// and of the log density after that step must not end the run.
struct ServicesOptimizeTwoScale : public testing::Test {
  ServicesOptimizeTwoScale()
      : init(init_ss),
        parameter(parameter_ss),
        model(context, 0, &model_ss),
        init_json("{\"x1\": 3e-9, \"x2\": 0}"),
        init_context(init_json) {}

  std::stringstream init_ss, parameter_ss, model_ss;
  stan::test::unit::instrumented_logger logger;
  stan::callbacks::stream_writer init;
  stan::test::unit::values_writer parameter;
  stan::io::empty_var_context context;
  stan_model model;
  std::stringstream init_json;
  stan::json::json_data init_context;
};

TEST_F(ServicesOptimizeTwoScale, lbfgs) {
  stan::test::unit::instrumented_interrupt interrupt;
  int return_code = stan::services::optimize::lbfgs<stan_model, false>(
      model, init_context, 0, 1, 0, 5, 0.001, 1e-12, 10000, 1e-8, 10000000,
      1e-8, 2000, false, 0, interrupt, logger, init, parameter);
  EXPECT_EQ(0, return_code);
  ASSERT_EQ(4, parameter.names_.size());
  EXPECT_NEAR(0, parameter.states_.back()[2], 1e-8);
  EXPECT_NEAR(5, parameter.states_.back()[3], 1e-4);
}

TEST_F(ServicesOptimizeTwoScale, bfgs) {
  stan::test::unit::instrumented_interrupt interrupt;
  int return_code = stan::services::optimize::bfgs<stan_model, false>(
      model, init_context, 0, 1, 0, 0.001, 1e-12, 10000, 1e-8, 10000000, 1e-8,
      2000, false, 0, interrupt, logger, init, parameter);
  EXPECT_EQ(0, return_code);
  ASSERT_EQ(4, parameter.names_.size());
  EXPECT_NEAR(0, parameter.states_.back()[2], 1e-8);
  EXPECT_NEAR(5, parameter.states_.back()[3], 1e-4);
}
