#include <stan/services/optimize/newton.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/simple_jacobian.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>

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

TEST_F(ServicesOptimize, withJacobian) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  int num_iterations = 1000;
  bool save_iterations = true;
  stan::test::unit::instrumented_interrupt interrupt;

  int return_code = stan::services::optimize::newton<stan_model, true>(
      model, context, seed, chain, init_radius, num_iterations, save_iterations,
      interrupt, logger, init, parameter);

  EXPECT_FLOAT_EQ(return_code, 0);

  ASSERT_EQ(2, parameter.names_.size());
  EXPECT_EQ("lp__", parameter.names_[0]);
  EXPECT_EQ("sigma", parameter.names_[1]);
  EXPECT_NEAR((3 + std::sqrt(13)) / 2, parameter.states_.back()[1], 0.001);
  EXPECT_GT(interrupt.call_count(), 0);
}

TEST_F(ServicesOptimize, withoutJacobian) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  int num_iterations = 1000;
  bool save_iterations = true;
  stan::test::unit::instrumented_interrupt interrupt;

  int return_code = stan::services::optimize::newton<stan_model, false>(
      model, context, seed, chain, init_radius, num_iterations, save_iterations,
      interrupt, logger, init, parameter);

  EXPECT_FLOAT_EQ(return_code, 0);

  ASSERT_EQ(2, parameter.names_.size());
  EXPECT_EQ("lp__", parameter.names_[0]);
  EXPECT_EQ("sigma", parameter.names_[1]);
  EXPECT_NEAR(3, parameter.states_.back()[1], 0.001);
  EXPECT_GT(interrupt.call_count(), 0);
}
