#include <stan/services/sample/fixed_param.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <iostream>

auto&& blah = stan::math::init_threadpool_tbb();

static constexpr size_t num_chains = 4;

class ServicesSamplesFixedParam : public testing::Test {
 public:
  ServicesSamplesFixedParam() : model(data_context, 0, &model_log) {
    for (int i = 0; i < num_chains; ++i) {
      init.push_back(stan::test::unit::instrumented_writer{});
      parameter.push_back(stan::test::unit::instrumented_writer{});
      diagnostic.push_back(stan::test::unit::instrumented_writer{});
      context.push_back(std::make_shared<stan::io::empty_var_context>());
    }
  }
  stan::io::empty_var_context data_context;
  std::stringstream model_log;
  stan_model model;
  stan::test::unit::instrumented_logger logger;
  std::vector<stan::test::unit::instrumented_writer> init;
  std::vector<stan::test::unit::instrumented_writer> parameter;
  std::vector<stan::test::unit::instrumented_writer> diagnostic;
  std::vector<std::shared_ptr<stan::io::empty_var_context>> context;
};

TEST_F(ServicesSamplesFixedParam, call_count) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int num_iterations = 10;

  int refresh = 0;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);

  int return_code = stan::services::sample::fixed_param(
      model, num_chains, context, seed, chain, init_radius, num_iterations, 1,
      refresh, interrupt, logger, init, parameter, diagnostic);
  EXPECT_EQ(0, return_code);

  // Expectations of message call counts
  EXPECT_EQ(num_iterations * num_chains, interrupt.call_count());
  for (int i = 0; i < num_chains; ++i) {
    EXPECT_EQ(1, parameter[i].call_count("vector_string"));
    EXPECT_EQ(num_iterations, parameter[i].call_count("vector_double"));
    EXPECT_EQ(1, diagnostic[i].call_count("vector_string"));
    EXPECT_EQ(num_iterations, diagnostic[i].call_count("vector_double"));
  }
}

TEST_F(ServicesSamplesFixedParam, output_sizes) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int num_iterations = 10;

  int refresh = 0;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);

  stan::services::sample::fixed_param(
      model, num_chains, context, seed, chain, init_radius, num_iterations, 1,
      refresh, interrupt, logger, init, parameter, diagnostic);

  for (int i = 0; i < num_chains; ++i) {
    std::vector<std::vector<std::string>> parameter_names
        = parameter[i].vector_string_values();
    std::vector<std::vector<double>> parameter_values
        = parameter[i].vector_double_values();
    std::vector<std::vector<std::string>> diagnostic_names
        = diagnostic[i].vector_string_values();
    std::vector<std::vector<double>> diagnostic_values
        = diagnostic[i].vector_double_values();

    // Expectations of parameter parameter names.
    ASSERT_EQ(4, parameter_names[0].size());
    EXPECT_EQ("lp__", parameter_names[0][0]);
    EXPECT_EQ("accept_stat__", parameter_names[0][1]);
    EXPECT_EQ("x", parameter_names[0][2]);
    EXPECT_EQ("y", parameter_names[0][3]);

    // Expect one name per parameter value.
    EXPECT_EQ(parameter_names[0].size(), parameter_values[0].size());
    EXPECT_EQ(diagnostic_names[0].size(), diagnostic_values[0].size());

    // Expect one vector of parameter values per iterations
    EXPECT_EQ(num_iterations, parameter_values.size());

    // Expect one call to set parameter names, and one set of output per
    // iteration.
    EXPECT_EQ("lp__", diagnostic_names[0][0]);
    EXPECT_EQ("accept_stat__", diagnostic_names[0][1]);
  }
}

TEST_F(ServicesSamplesFixedParam, parameter_checks) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int num_iterations = 10;

  int refresh = 0;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);

  int return_code = stan::services::sample::fixed_param(
      model, num_chains, context, seed, chain, init_radius, num_iterations, 1,
      refresh, interrupt, logger, init, parameter, diagnostic);
  for (int i = 0; i < num_chains; ++i) {
    std::vector<std::vector<std::string>> parameter_names
        = parameter[i].vector_string_values();
    std::vector<std::vector<double>> parameter_values
        = parameter[i].vector_double_values();
    std::vector<std::vector<std::string>> diagnostic_names
        = diagnostic[i].vector_string_values();
    std::vector<std::vector<double>> diagnostic_values
        = diagnostic[i].vector_double_values();

    // Expect parameter values to stay at zero.
    EXPECT_DOUBLE_EQ(0.0, parameter_values.front()[1])
        << "initial memory_writer should be (0, 0)";
    EXPECT_DOUBLE_EQ(0.0, parameter_values.front()[2])
        << "initial memory_writer should be (0, 0)";
    EXPECT_DOUBLE_EQ(0.0, parameter_values.back()[1])
        << "final memory_writer should be (0, 0)";
    EXPECT_DOUBLE_EQ(0.0, parameter_values.back()[2])
        << "final memory_writer should be (0, 0)";
  }
  EXPECT_EQ(return_code, 0);
}

TEST_F(ServicesSamplesFixedParam, output_regression) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int num_iterations = 10;

  int refresh = 0;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);

  stan::services::sample::fixed_param(
      model, num_chains, context, seed, chain, init_radius, num_iterations, 1,
      refresh, interrupt, logger, init, parameter, diagnostic);
  for (int i = 0; i < num_chains; ++i) {
    std::vector<std::vector<std::string>> parameter_names
        = parameter[i].vector_string_values();
    std::vector<std::vector<double>> parameter_values
        = parameter[i].vector_double_values();
    std::vector<std::vector<std::string>> diagnostic_names
        = diagnostic[i].vector_string_values();
    std::vector<std::vector<double>> diagnostic_values
        = diagnostic[i].vector_double_values();
    std::vector<std::string> init_values = init[i].string_values();

    EXPECT_EQ(num_chains, logger.find_info("Elapsed Time:"));
    EXPECT_EQ(num_chains, logger.find_info("seconds (Warm-up)"));
    EXPECT_EQ(num_chains, logger.find_info("seconds (Sampling)"));
    EXPECT_EQ(num_chains, logger.find_info("seconds (Total)"));
    EXPECT_EQ(0, logger.call_count_error());

    EXPECT_EQ(0, init_values.size());
  }
}
