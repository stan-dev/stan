#include <stan/services/sample/fixed_param.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <iostream>


class ServicesSamplesFixedParam : public testing::Test {
public:
  ServicesSamplesFixedParam()
    : model(context, &model_log) {}

  std::stringstream model_log;
  stan::test::unit::instrumented_writer message, init, error;
  stan::test::unit::instrumented_writer parameter, diagnostic;
  stan::io::empty_var_context context;
  stan_model model;
};


TEST_F(ServicesSamplesFixedParam, call_count) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int num_iterations = 10;

  int refresh = 0;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);
  
  int return_code = stan::services::sample::fixed_param(model, context,
                                                        seed, chain, init_radius,
                                                        num_iterations,
                                                        1,
                                                        refresh,
                                                        interrupt,
                                                        message,
                                                        error,
                                                        init,
                                                        parameter,
                                                        diagnostic);
  EXPECT_EQ(0, return_code);

  std::vector<std::vector<std::string> > parameter_names;
  parameter_names = parameter.vector_string_values();
  std::vector<std::vector<double> > parameter_values;
  parameter_values = parameter.vector_double_values();
  std::vector<std::vector<std::string> > diagnostic_names;
  diagnostic_names = diagnostic.vector_string_values();
  std::vector<std::vector<double> > diagnostic_values;
  diagnostic_values = diagnostic.vector_double_values();

  // Expecatations of message call counts
  EXPECT_EQ(num_iterations, interrupt.call_count());
  EXPECT_EQ(1, parameter.call_count("vector_string"));
  EXPECT_EQ(num_iterations, parameter.call_count("vector_double"));
  EXPECT_EQ(1, diagnostic.call_count("vector_string"));
  EXPECT_EQ(num_iterations, diagnostic.call_count("vector_double"));
}


TEST_F(ServicesSamplesFixedParam, output_sizes) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int num_iterations = 10;

  int refresh = 0;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);
  
  stan::services::sample::fixed_param(model, context,
                                      seed, chain, init_radius,
                                      num_iterations,
                                      1,
                                      refresh,
                                      interrupt,
                                      message,
                                      error,
                                      init,
                                      parameter,
                                      diagnostic);

  std::vector<std::vector<std::string> > parameter_names;
  parameter_names = parameter.vector_string_values();
  std::vector<std::vector<double> > parameter_values;
  parameter_values = parameter.vector_double_values();
  std::vector<std::vector<std::string> > diagnostic_names;
  diagnostic_names = diagnostic.vector_string_values();
  std::vector<std::vector<double> > diagnostic_values;
  diagnostic_values = diagnostic.vector_double_values();

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


TEST_F(ServicesSamplesFixedParam, parameter_checks) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int num_iterations = 10;

  int refresh = 0;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);
  
  int return_code = stan::services::sample::fixed_param(model, context,
                                                        seed, chain, init_radius,
                                                        num_iterations,
                                                        1,
                                                        refresh,
                                                        interrupt,
                                                        message,
                                                        error,
                                                        init,
                                                        parameter,
                                                        diagnostic);

  std::vector<std::vector<std::string> > parameter_names;
  parameter_names = parameter.vector_string_values();
  std::vector<std::vector<double> > parameter_values;
  parameter_values = parameter.vector_double_values();
  std::vector<std::vector<std::string> > diagnostic_names;
  diagnostic_names = diagnostic.vector_string_values();
  std::vector<std::vector<double> > diagnostic_values;
  diagnostic_values = diagnostic.vector_double_values();

  // Expect parameter values to stay at zero.
  EXPECT_DOUBLE_EQ(0.0, parameter_values.front()[1])
    << "initial memory_writer should be (0, 0)";
  EXPECT_DOUBLE_EQ(0.0, parameter_values.front()[2])
    << "initial memory_writer should be (0, 0)";
  EXPECT_DOUBLE_EQ(0.0, parameter_values.back()[1])
    << "final memory_writer should be (0, 0)";
  EXPECT_DOUBLE_EQ(0.0, parameter_values.back()[2])
    << "final memory_writer should be (0, 0)";
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
  
  stan::services::sample::fixed_param(model, context,
                                      seed, chain, init_radius,
                                      num_iterations,
                                      1,
                                      refresh,
                                      interrupt,
                                      message,
                                      error,
                                      init,
                                      parameter,
                                      diagnostic);

  std::vector<std::vector<std::string> > parameter_names;
  parameter_names = parameter.vector_string_values();
  std::vector<std::vector<double> > parameter_values;
  parameter_values = parameter.vector_double_values();
  std::vector<std::vector<std::string> > diagnostic_names;
  diagnostic_names = diagnostic.vector_string_values();
  std::vector<std::vector<double> > diagnostic_values;
  diagnostic_values = diagnostic.vector_double_values();
  std::vector<std::string> message_values;
  message_values = message.string_values();
  std::vector<std::string> init_values;
  init_values = init.string_values();
  std::vector<std::string> error_values;
  error_values = error.string_values();

  EXPECT_NE(std::string::npos, message_values[0].find("Elapsed Time:"));
  EXPECT_NE(std::string::npos, message_values[0].find("seconds (Warm-up)"));
  EXPECT_NE(std::string::npos, message_values[1].find("seconds (Sampling)"));
  EXPECT_NE(std::string::npos, message_values[2].find("seconds (Total)"));

  EXPECT_EQ(0, init_values.size());
  EXPECT_EQ(0, error_values.size());
}
