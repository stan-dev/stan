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


TEST_F(ServicesSamplesFixedParam, rosenbrock) {
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
                                                   init,
                                                   error,
                                                   parameter,
                                                   diagnostic);

  // Expect interrupt to be called once per iteration.
  EXPECT_EQ(num_iterations, interrupt.call_count());

  // Expecatations of message call counts
  // Expectations of messages

  EXPECT_EQ("\n Elapsed Time: 0 seconds (Warm-up)\n", 
    message_ss.str().substr(0,36));

  EXPECT_EQ("0,0\n", init_ss.str());

  // Expect on call to set parameter names, and one set of output per
  // iteration.
  EXPECT_EQ(parameter.call_count("vector_string"), 1);
  EXPECT_EQ(parameter.call_count("vector_double"), num_iterations);

  // Expectations of parameter parameter names.  
  std::vector<std::vector<std::string> > parameter_names;
  parameter_names = parameter.vector_string_values();
  ASSERT_EQ(4, parameter_names_.size());
  EXPECT_EQ("lp__", parameter_names[0][0]);
  EXPECT_EQ("accept_stat__", parameter_names[0][1]);
  EXPECT_EQ("x", parameter_names[0][2]);
  EXPECT_EQ("y", parameter_names[0][3]);

  std::vector<std::vector<double> > parameter_values;
  parameter_values = parameter.vector_double_values();

  // Expect one parameter name per parameter value.
  EXPECT_EQ(parameter_names[0].size(), parameter_values[0].size());

  // Expect one vector of parameter values per iterations
  EXPECT_EQ(num_iterations, parameter_values.size());
 
  // Expect one call to set parameter names, and one set of output per
  // iteration, not sure where the "+1" is coming from yet...
  EXPECT_EQ(diagnostic.call_count("vector_string"), 1);
  EXPECT_EQ(diagnostic.call_count("vector_double"), num_iterations + 1);
  std::vector<std::vector<std::string> > diagnostic_names;
  diagnostic_names = diagnostic.vector_string_values();
  EXPECT_EQ(diagnostic_names[0][0], "lp__");
  EXPECT_EQ(diagnostic_names[0][1], "accept_stat__");
  std::vector<std::vector<double> > diagnostic_values;
  diagnostic_values = diagnostic.vector_double_values();
  EXPECT_EQ(diagnostic_names[0].size(), diagnostic_values[0].size());

  // Expect parameter values to stay at zero.
  EXPECT_DOUBLE_EQ(0.0, parameter_values.front()[1])
    << "initial memory_writer should be (0, 0)";
  EXPECT_DOUBLE_EQ(0.0, parameter_values.front()[2])
    << "initial memory_writer should be (0, 0)";
  EXPECT_DOUBLE_EQ(0.0, parameter_values.back()[1])
    << "final memory_writer should be (0, 0)";
  EXPECT_DOUBLE_EQ(0.0, parameter_values.back()[2])
    << "final memory_writer should be (0, 0)";
  EXPECT_DOUBLE_EQ(return_code, 0);


}




