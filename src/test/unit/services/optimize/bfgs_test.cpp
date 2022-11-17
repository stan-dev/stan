#include <stan/services/optimize/bfgs.hpp>
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

  bool save_iterations = true;
  int refresh = 0;
  stan::test::unit::instrumented_interrupt interrupt;

  int return_code = stan::services::optimize::bfgs(
      model, context, seed, chain, init_radius, 0.001, 1e-12, 10000, 1e-8,
      10000000, 1e-8, 2000, save_iterations, refresh, interrupt, logger, init,
      parameter);

  EXPECT_EQ(logger.call_count(), logger.call_count_info())
      << "all output to info";
  EXPECT_EQ(1, logger.find("Initial log joint probability = -1"));
  EXPECT_EQ(1, logger.find("Optimization terminated normally: "));
  EXPECT_EQ(1, logger.find("  Convergence detected: relative gradient "
                           "magnitude is below tolerance"));

  EXPECT_EQ("0,0\n", init_ss.str());

  ASSERT_EQ(3, parameter.names_.size());
  EXPECT_EQ("lp__", parameter.names_[0]);
  EXPECT_EQ("x", parameter.names_[1]);
  EXPECT_EQ("y", parameter.names_[2]);

  EXPECT_EQ(20, parameter.states_.size());
  EXPECT_FLOAT_EQ(0, parameter.states_.front()[1])
      << "initial value should be (0, 0)";
  EXPECT_FLOAT_EQ(0, parameter.states_.front()[2])
      << "initial value should be (0, 0)";
  EXPECT_FLOAT_EQ(1, parameter.states_.back()[1])
      << "optimal value should be (1, 1)";
  EXPECT_FLOAT_EQ(1, parameter.states_.back()[2])
      << "optimal value should be (1, 1)";
  EXPECT_FLOAT_EQ(return_code, 0);
  EXPECT_EQ(19, interrupt.call_count());
}
