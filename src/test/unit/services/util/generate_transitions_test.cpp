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
  stan::test::unit::instrumented_writer init;
  stan::test::unit::instrumented_writer parameter, diagnostic;
  stan::test::unit::instrumented_logger logger;
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
                                       logger, diagnostic);

  stan::mcmc::fixed_param_sampler sampler;
  stan::services::util::mcmc_writer
    writer(parameter, diagnostic, logger);
  Eigen::VectorXd cont_params(cont_vector.size());
  for (size_t i = 0; i < cont_vector.size(); i++)
    cont_params[i] = cont_vector[i];
  stan::mcmc::sample s(cont_params, 0, 0);

  writer.write_sample_names(s, sampler, model);
  writer.write_diagnostic_names(s, sampler, model);

  stan::services::util::generate_transitions(
    sampler, num_iterations, 0, 20, 1, refresh, true, false, writer,
    s, model, rng, interrupt, logger);

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

  // Expect no messages and no init messages
  EXPECT_EQ(logger.call_count(), 0);
  EXPECT_EQ(init.call_count(), 0);

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
                                       logger, diagnostic);

  stan::mcmc::fixed_param_sampler sampler;
  stan::services::util::mcmc_writer
    writer(parameter, diagnostic, logger);
  Eigen::VectorXd cont_params(cont_vector.size());
  for (size_t i = 0; i < cont_vector.size(); i++)
    cont_params[i] = cont_vector[i];
  stan::mcmc::sample s(cont_params, 0, 0);

  writer.write_sample_names(s, sampler, model);
  writer.write_diagnostic_names(s, sampler, model);

  stan::services::util::generate_transitions(
    sampler, num_iterations, 0, 20, 1, refresh, true, false, writer,
    s, model, rng, interrupt, logger);

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

TEST_F(ServicesSamplesGenerateTransitions, messages) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int refresh = 2;
  int num_iterations = 10;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);

  boost::ecuyer1988 rng = stan::services::util::create_rng(seed, chain);

  std::vector<int> disc_vector;
  std::vector<double> cont_vector
    = stan::services::util::initialize(model, context, rng, init_radius,
                                       false,
                                       logger, diagnostic);

  stan::mcmc::fixed_param_sampler sampler;
  stan::services::util::mcmc_writer
    writer(parameter, diagnostic, logger);
  Eigen::VectorXd cont_params(cont_vector.size());
  for (size_t i = 0; i < cont_vector.size(); i++)
    cont_params[i] = cont_vector[i];
  stan::mcmc::sample s(cont_params, 0, 0);

  writer.write_sample_names(s, sampler, model);
  writer.write_diagnostic_names(s, sampler, model);

  stan::services::util::generate_transitions(
    sampler, num_iterations, 0, 20, 1, refresh, true, true, writer,
    s, model, rng, interrupt, logger);

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

  EXPECT_TRUE(logger.find_info("Iteration:  1 / 20 [  5%]  (Warmup)"));
  EXPECT_TRUE(logger.find_info("Iteration:  2 / 20 [ 10%]  (Warmup)"));
  EXPECT_TRUE(logger.find_info("Iteration:  4 / 20 [ 20%]  (Warmup)"));
  EXPECT_TRUE(logger.find_info("Iteration:  6 / 20 [ 30%]  (Warmup)"));
  EXPECT_TRUE(logger.find_info("Iteration:  8 / 20 [ 40%]  (Warmup)"));
  EXPECT_TRUE(logger.find_info("Iteration: 10 / 20 [ 50%]  (Warmup)"));

  EXPECT_FALSE(logger.find_info("Iteration: 3 / 20 [ 15%]  (Warmup)"));
  EXPECT_FALSE(logger.find_info("Iteration: 5 / 20 [ 25%]  (Warmup)"));
  EXPECT_FALSE(logger.find_info("Iteration: 7 / 20 [ 35%]  (Warmup)"));
  EXPECT_FALSE(logger.find_info("Iteration: 9 / 20 [ 45%]  (Warmup)"));

  stan::test::unit::instrumented_logger logger2;
  stan::services::util::generate_transitions(
    sampler, num_iterations, 10, 20, 1, refresh, true, false, writer,
    s, model, rng, interrupt, logger2);

  EXPECT_TRUE(logger2.find_info("Iteration: 11 / 20 [ 55%]  (Sampling)"));
  EXPECT_TRUE(logger2.find_info("Iteration: 12 / 20 [ 60%]  (Sampling)"));
  EXPECT_TRUE(logger2.find_info("Iteration: 14 / 20 [ 70%]  (Sampling)"));
  EXPECT_TRUE(logger2.find_info("Iteration: 16 / 20 [ 80%]  (Sampling)"));
  EXPECT_TRUE(logger2.find_info("Iteration: 18 / 20 [ 90%]  (Sampling)"));
  EXPECT_TRUE(logger2.find_info("Iteration: 20 / 20 [100%]  (Sampling)"));

  EXPECT_FALSE(logger2.find_info("Iteration: 13 / 20 [ 65%]  (Sampling)"));
  EXPECT_FALSE(logger2.find_info("Iteration: 15 / 20 [ 75%]  (Sampling)"));
  EXPECT_FALSE(logger2.find_info("Iteration: 17 / 20 [ 85%]  (Sampling)"));
  EXPECT_FALSE(logger2.find_info("Iteration: 19 / 20 [ 95%]  (Sampling)"));
}

TEST_F(ServicesSamplesGenerateTransitions, iteration_messages) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int num_iterations = 10;
  stan::test::unit::instrumented_interrupt interrupt;
  stan::test::unit::instrumented_iteration iteration;
  boost::ecuyer1988 rng = stan::services::util::create_rng(seed, chain);

  std::vector<int> disc_vector;
  std::vector<double> cont_vector
    = stan::services::util::initialize(model, context, rng, init_radius,
                                       false,
                                       logger, diagnostic);

  stan::mcmc::fixed_param_sampler sampler;
  stan::services::util::mcmc_writer
    writer(parameter, diagnostic, logger);
  Eigen::VectorXd cont_params(cont_vector.size());
  for (size_t i = 0; i < cont_vector.size(); i++)
    cont_params[i] = cont_vector[i];
  stan::mcmc::sample s(cont_params, 0, 0);

  writer.write_sample_names(s, sampler, model);
  writer.write_diagnostic_names(s, sampler, model);

  stan::services::util::generate_transitions(
    sampler, num_iterations, 0, 20, 1, true, true, writer,
    s, model, rng, iteration, interrupt, logger);

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

  EXPECT_EQ(num_iterations, iteration.call_count());


  stan::services::util::generate_transitions(
    sampler, num_iterations, 10, 20, 1, true, false, writer,
    s, model, rng, iteration, interrupt, logger);
  EXPECT_EQ(20, iteration.call_count());


  EXPECT_FALSE(logger.find_info("Iteration:  1 / 20 [  5%]  (Warmup)"));
}
