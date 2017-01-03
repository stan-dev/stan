#include <stan/services/util/generate_transitions.hpp>
#include <stan/services/sample/fixed_param.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/create_rng.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <iostream>
#include <exception>

class ServicesSamplesGenerateTransitions : public testing::Test {
public:
  ServicesSamplesGenerateTransitions()
    : model(context, &model_log) {}

  std::stringstream model_log;
  stan::test::unit::instrumented_writer message, init, error;
  stan::test::unit::instrumented_writer parameter, diagnostic;
  stan::io::empty_var_context context;
  stan_model model;
};


TEST_F(ServicesSamplesGenerateTransitions, call_counting) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int refresh = 0;
  int num_iterations = 10;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);

  boost::ecuyer1988 rng = stan::services::util::create_rng(seed, chain);

  std::vector<int> disc_vector;
  std::vector<double> cont_vector
    = stan::services::util::initialize(model, context, rng, init_radius,
                                       false,
                                       message, diagnostic);

  stan::mcmc::fixed_param_sampler sampler;
  stan::services::util::mcmc_writer
    writer(parameter, diagnostic, message);
  Eigen::VectorXd cont_params(cont_vector.size());
  for (size_t i = 0; i < cont_vector.size(); i++)
    cont_params[i] = cont_vector[i];
  stan::mcmc::sample s(cont_params, 0, 0);

  writer.write_sample_names(s, sampler, model);
  writer.write_diagnostic_names(s, sampler, model);

  stan::services::util::generate_transitions(
    sampler, num_iterations, 0, 20, 1, refresh, true, false, writer,
    s, model, rng, interrupt, message, error);

  std::vector<std::vector<std::string> > parameter_names;
  parameter_names = parameter.vector_string_values();
  std::vector<std::vector<double> > parameter_values;
  parameter_values = parameter.vector_double_values();
  std::vector<std::vector<std::string> > diagnostic_names;
  diagnostic_names = diagnostic.vector_string_values();
  std::vector<std::vector<double> > diagnostic_values;
  diagnostic_values = diagnostic.vector_double_values();

  // Expect interrupt to be called once per iteration.
  EXPECT_EQ(interrupt.call_count(), num_iterations);

  // Expect no messages, no init messages, and no error messages.
  EXPECT_EQ(message.call_count(), 0);
  EXPECT_EQ(init.call_count(), 0);
  EXPECT_EQ(error.call_count(), 0);

  // Expect on call to set parameter names, and one set of output per
  // iteration.
  EXPECT_EQ(parameter.call_count("vector_string"), 1);
  EXPECT_EQ(parameter.call_count("vector_double"), num_iterations);


  // Expect one call to set parameter names, and one set of output per
  // iteration, not sure where the "+1" is coming from yet...
  EXPECT_EQ(diagnostic.call_count("vector_string"), 1);
  EXPECT_EQ(diagnostic.call_count("vector_double"), num_iterations + 1);


}


TEST_F(ServicesSamplesGenerateTransitions, output_sizes) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int refresh = 0;
  int num_iterations = 10;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);

  boost::ecuyer1988 rng = stan::services::util::create_rng(seed, chain);

  std::vector<int> disc_vector;
  std::vector<double> cont_vector
    = stan::services::util::initialize(model, context, rng, init_radius,
                                       false,
                                       message, diagnostic);

  stan::mcmc::fixed_param_sampler sampler;
  stan::services::util::mcmc_writer
    writer(parameter, diagnostic, message);
  Eigen::VectorXd cont_params(cont_vector.size());
  for (size_t i = 0; i < cont_vector.size(); i++)
    cont_params[i] = cont_vector[i];
  stan::mcmc::sample s(cont_params, 0, 0);

  writer.write_sample_names(s, sampler, model);
  writer.write_diagnostic_names(s, sampler, model);

  stan::services::util::generate_transitions(
    sampler, num_iterations, 0, 20, 1, refresh, true, false, writer,
    s, model, rng, interrupt, message, error);

  std::vector<std::vector<std::string> > parameter_names;
  parameter_names = parameter.vector_string_values();
  std::vector<std::vector<double> > parameter_values;
  parameter_values = parameter.vector_double_values();
  std::vector<std::vector<std::string> > diagnostic_names;
  diagnostic_names = diagnostic.vector_string_values();
  std::vector<std::vector<double> > diagnostic_values;
  diagnostic_values = diagnostic.vector_double_values();

  // First parameters are log density and acceptance prob.
  EXPECT_EQ(parameter_names[0][0], "lp__");
  EXPECT_EQ(parameter_names[0][1], "accept_stat__");

  // First diagnostic parameters are log density and acceptance
  // prob.
  EXPECT_EQ(diagnostic_names[0][0], "lp__");
  EXPECT_EQ(diagnostic_names[0][1], "accept_stat__");

  // Expect one parameter name per parameter value.
  EXPECT_EQ(parameter_names[0].size(), parameter_values[0].size());
  EXPECT_EQ(diagnostic_names[0].size(), diagnostic_values[0].size());

}
