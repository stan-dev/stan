#include <stan/services/optimize/newton.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
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

TEST_F(ServicesOptimize, rosenbrock) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  int num_iterations = 1000;
  bool save_iterations = true;
  stan::test::unit::instrumented_interrupt interrupt;

  int return_code = stan::services::optimize::newton(
      model, context, seed, chain, init_radius, num_iterations, save_iterations,
      interrupt, logger, init, parameter);

  EXPECT_EQ(0, return_code);
  EXPECT_EQ(logger.call_count(), logger.call_count_info())
      << "all output to info";
  EXPECT_EQ(1, logger.find("Initial log joint probability = -1"));
  EXPECT_EQ(1, logger.find("Iteration  1. Log joint probability ="));

  ASSERT_EQ(3, parameter.names_.size());
  EXPECT_EQ("lp__", parameter.names_[0]);
  EXPECT_EQ("x", parameter.names_[1]);
  EXPECT_EQ("y", parameter.names_[2]);

  EXPECT_GT(parameter.states_.size(), 0);
  EXPECT_FLOAT_EQ(0, parameter.states_.front()[1])
      << "initial value should be (0, 0)";
  EXPECT_FLOAT_EQ(0, parameter.states_.front()[2])
      << "initial value should be (0, 0)";
  EXPECT_NEAR(1, parameter.states_.back()[1], 1e-3)
      << "optimal value should be (1, 1)";
  EXPECT_NEAR(1, parameter.states_.back()[2], 1e-3)
      << "optimal value should be (1, 1)";
  EXPECT_FLOAT_EQ(return_code, 0);
  EXPECT_LT(0, interrupt.call_count());
}

TEST_F(ServicesOptimize, rosenbrock_no_save_iterations) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  int num_iterations = 1000;
  bool save_iterations = false;
  stan::test::unit::instrumented_interrupt interrupt;

  int return_code = stan::services::optimize::newton(
      model, context, seed, chain, init_radius, num_iterations, save_iterations,
      interrupt, logger, init, parameter);

  EXPECT_EQ(0, return_code);
  EXPECT_EQ(logger.call_count(), logger.call_count_info())
      << "all output to info";
  EXPECT_EQ(1, logger.find("Initial log joint probability = -1"));
  EXPECT_EQ(1, logger.find("Iteration  1. Log joint probability ="));

  EXPECT_EQ("0,0\n", init_ss.str());

  ASSERT_EQ(3, parameter.names_.size());
  EXPECT_EQ("lp__", parameter.names_[0]);
  EXPECT_EQ("x", parameter.names_[1]);
  EXPECT_EQ("y", parameter.names_[2]);

  EXPECT_EQ(1, parameter.states_.size());
  EXPECT_NEAR(1, parameter.states_.back()[1], 1e-3)
      << "optimal value should be (1, 1)";
  EXPECT_NEAR(1, parameter.states_.back()[2], 1e-3)
      << "optimal value should be (1, 1)";
  EXPECT_FLOAT_EQ(return_code, 0);
  EXPECT_LT(0, interrupt.call_count());
}
