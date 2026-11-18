#include <stan/services/optimize/lbfgs.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/json/json_data.hpp>
#include <test/test-models/good/optimization/simple_jacobian.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <cmath>

struct ServicesOptimize : public testing::Test {
  ServicesOptimize()
      : init(init_ss), parameter(parameter_ss), model(context, 0, &model_ss) {}

  std::stringstream init_ss, parameter_ss, model_ss;
  stan::test::unit::instrumented_logger logger;
  stan::callbacks::stream_writer init;
  stan::test::unit::values_writer parameter;
  stan::io::empty_var_context context;
  stan_model model;
};

TEST_F(ServicesOptimize, with_jacobian) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  bool save_iterations = true;
  int refresh = 0;
  stan::test::unit::instrumented_interrupt interrupt;

  int return_code = stan::services::optimize::lbfgs<stan_model, true>(
      model, context, seed, chain, init_radius, 5, 0.001, 1e-12, 10000, 1e-8,
      10000000, 1e-8, 2000, save_iterations, refresh, interrupt, logger, init,
      parameter);

  EXPECT_TRUE(logger.find("Optimization terminated normally: "));
  EXPECT_FLOAT_EQ(return_code, 0);

  ASSERT_EQ(3, parameter.names_.size());
  EXPECT_EQ("lp__", parameter.names_[0]);
  EXPECT_EQ("converged__", parameter.names_[1]);
  EXPECT_EQ("sigma", parameter.names_[2]);
  EXPECT_NEAR((3 + std::sqrt(13)) / 2, parameter.states_.back()[2], 0.0001);
}

TEST_F(ServicesOptimize, without_jacobian) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  bool save_iterations = true;
  int refresh = 0;
  stan::test::unit::instrumented_interrupt interrupt;

  int return_code = stan::services::optimize::lbfgs<stan_model, false>(
      model, context, seed, chain, init_radius, 5, 0.001, 1e-12, 10000, 1e-8,
      10000000, 1e-8, 2000, save_iterations, refresh, interrupt, logger, init,
      parameter);

  EXPECT_TRUE(logger.find("Optimization terminated normally: "));
  EXPECT_FLOAT_EQ(return_code, 0);

  ASSERT_EQ(3, parameter.names_.size());
  EXPECT_EQ("lp__", parameter.names_[0]);
  EXPECT_EQ("converged__", parameter.names_[1]);
  EXPECT_EQ("sigma", parameter.names_[2]);
  EXPECT_NEAR(3, parameter.states_.back()[2], 0.0001);
}

TEST_F(ServicesOptimize, lbfgs_large_initial_gradient) {
  // At sigma = exp(20) the gradient in log(sigma) is (sigma - 3) * sigma =
  // 2.4e17. The first trial step was init_alpha times that gradient, the
  // step that is needed is below minAlpha = 1e-12, and the first line search
  // failed (error code 70). The first trial step now moves log(sigma) by at
  // most 1.
  std::stringstream init_json("{\"sigma\": 485165195.40979}");
  stan::json::json_data init_context(init_json);
  stan::test::unit::instrumented_interrupt interrupt;
  int return_code = stan::services::optimize::lbfgs<stan_model, false>(
      model, init_context, 0, 1, 0, 5, 0.001, 1e-12, 10000, 1e-8, 10000000,
      1e-8, 2000, false, 0, interrupt, logger, init, parameter);
  EXPECT_EQ(0, return_code);
  EXPECT_NEAR(3, parameter.states_.back()[2], 0.0001);
}
