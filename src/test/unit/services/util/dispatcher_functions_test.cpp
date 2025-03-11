#include <stan/callbacks/dispatcher.hpp>
#include <stan/services/util/dispatcher_functions.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/configure_dispatcher.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <iostream>
#include <memory>
#include <sstream>

class DispatcherFunctionsTest : public testing::Test {
 public:
  DispatcherFunctionsTest() : model(context, 0, &model_log) {}

  std::stringstream model_log;
  stan::io::empty_var_context context;
  stan_model model;

  stan::test::unit::instrumented_logger logger;
  stan::callbacks::interrupt interrupt;

  // Streams to capture dispatcher output
  std::shared_ptr<std::stringstream> sample_stream;
  std::shared_ptr<std::stringstream> diagnostic_stream;
  std::shared_ptr<std::stringstream> metric_stream;
  std::shared_ptr<std::stringstream> init_stream;

  stan::callbacks::dispatcher dispatcher;

  void SetUp() override {
    // Create stringstreams
    sample_stream = std::make_shared<std::stringstream>();
    diagnostic_stream = std::make_shared<std::stringstream>();
    metric_stream = std::make_shared<std::stringstream>();
    init_stream = std::make_shared<std::stringstream>();

    // Initialize the dispatcher
    std::unordered_map<stan::callbacks::info_type,
                       std::shared_ptr<std::ostream>,
                       stan::callbacks::info_type_hash>
        output_streams;

    output_streams[stan::callbacks::info_type::SAMPLE] = sample_stream;
    output_streams[stan::callbacks::info_type::DIAGNOSTIC] = diagnostic_stream;
    output_streams[stan::callbacks::info_type::METRIC] = metric_stream;
    output_streams[stan::callbacks::info_type::UNCONSTRAINED_INITS]
        = init_stream;

    dispatcher
        = stan::services::util::configure_dispatcher(std::move(output_streams));
  }
};

TEST_F(DispatcherFunctionsTest, WriteSampleHeader) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  stan::rng_t rng = stan::services::util::create_rng(seed, chain);

  std::vector<double> cont_vector = stan::services::util::initialize(
      model, context, rng, init_radius, false, logger, dispatcher);

  stan::mcmc::adapt_diag_e_nuts<stan_model, stan::rng_t> sampler(model, rng);
  Eigen::VectorXd cont_params(cont_vector.size());
  for (size_t i = 0; i < cont_vector.size(); i++)
    cont_params[i] = cont_vector[i];
  stan::mcmc::sample s(cont_params, 0, 0);

  size_t num_model_values = 0;
  stan::services::util::write_sample_header(s, sampler, model, dispatcher,
                                            num_model_values);

  // Check that header was written to sample stream
  std::string output = sample_stream->str();

  // Should contain parameter names
  EXPECT_TRUE(output.find("lp__") != std::string::npos);
  EXPECT_TRUE(output.find("accept_stat__") != std::string::npos);
  EXPECT_TRUE(output.find("stepsize__") != std::string::npos);
  EXPECT_TRUE(output.find("treedepth__") != std::string::npos);
  EXPECT_TRUE(output.find("x") != std::string::npos);
  EXPECT_TRUE(output.find("y") != std::string::npos);

  // Check that num_model_values was set correctly (2 for Rosenbrock x,y)
  EXPECT_EQ(num_model_values, 2);
}

TEST_F(DispatcherFunctionsTest, WriteSample) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  stan::rng_t rng = stan::services::util::create_rng(seed, chain);

  std::vector<double> cont_vector = stan::services::util::initialize(
      model, context, rng, init_radius, false, logger, dispatcher);

  stan::mcmc::adapt_diag_e_nuts<stan_model, stan::rng_t> sampler(model, rng);
  Eigen::VectorXd cont_params(cont_vector.size());
  for (size_t i = 0; i < cont_vector.size(); i++)
    cont_params[i] = cont_vector[i];
  stan::mcmc::sample s(cont_params, 0, 0);

  // First write the header to get num_model_values
  size_t num_model_values = 0;
  stan::services::util::write_sample_header(s, sampler, model, dispatcher,
                                            num_model_values);

  std::string header = sample_stream->str();
  std::string expect(
      "lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,"
      "energy__,x,y\n");
  EXPECT_EQ(header, expect);

  // Clear the stream for the next test
  sample_stream->str("");
  sample_stream->clear();

  // Now write a sample
  stan::services::util::write_sample(rng, s, sampler, model, dispatcher, logger,
                                     num_model_values);

  // Check the output
  std::string output = sample_stream->str();

  // Parse the CSV output to verify the values
  std::stringstream ss(output);
  std::string line;
  std::getline(ss, line);

  // Make sure there are values in the output
  EXPECT_FALSE(line.empty());

  // Confirm it contains all the expected values in CSV format
  int comma_count = 0;
  for (char c : line) {
    if (c == ',')
      comma_count++;
  }
  EXPECT_EQ(comma_count, 8);
}

TEST_F(DispatcherFunctionsTest, WriteDiagnostic) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  stan::rng_t rng = stan::services::util::create_rng(seed, chain);

  std::vector<double> cont_vector = stan::services::util::initialize(
      model, context, rng, init_radius, false, logger, dispatcher);

  stan::mcmc::adapt_diag_e_nuts<stan_model, stan::rng_t> sampler(model, rng);
  Eigen::VectorXd cont_params(cont_vector.size());
  for (size_t i = 0; i < cont_vector.size(); i++)
    cont_params[i] = cont_vector[i];
  stan::mcmc::sample s(cont_params, 0, 0);

  // First write the header to get num_model_values
  size_t num_model_values = 0;
  stan::services::util::write_diagnostics_header(s, sampler, model, dispatcher);

  std::string header = diagnostic_stream->str();
  std::string expect(
      "lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,"
      "energy__,x,y,p_x,p_y,g_x,g_y\n");
  EXPECT_EQ(header, expect);

  // Clear the stream for the next test
  diagnostic_stream->str("");
  diagnostic_stream->clear();

  // Now write a sample
  stan::services::util::write_diagnostics(s, sampler, dispatcher);

  // Check the output
  std::string output = diagnostic_stream->str();

  // Parse the CSV output to verify the values
  std::stringstream ss(output);
  std::string line;
  std::getline(ss, line);

  // Make sure there are values in the output
  EXPECT_FALSE(line.empty());

  // Confirm it contains all the expected values in CSV format
  int comma_count = 0;
  for (char c : line) {
    if (c == ',')
      comma_count++;
  }
  EXPECT_EQ(comma_count, 12);
}

TEST_F(DispatcherFunctionsTest, WriteAdaptFinish) {
  stan::services::util::write_adapt_finish(dispatcher);

  // Check that message was written to sample stream
  std::string output = sample_stream->str();
  EXPECT_TRUE(output.find("Adaptation terminated") != std::string::npos);
}

TEST_F(DispatcherFunctionsTest, WriteTiming) {
  double warmup_time = 10.5;
  double sampling_time = 20.3;

  stan::services::util::write_timing(warmup_time, sampling_time, dispatcher,
                                     logger);

  // Check sample stream
  std::string sample_output = sample_stream->str();
  EXPECT_TRUE(sample_output.find("10.5 seconds (Warm-up)")
              != std::string::npos);
  EXPECT_TRUE(sample_output.find("20.3 seconds (Sampling)")
              != std::string::npos);
  EXPECT_TRUE(sample_output.find("30.8 seconds (Total)") != std::string::npos);

  // Check diagnostic stream
  std::string diag_output = diagnostic_stream->str();
  EXPECT_TRUE(diag_output.find("10.5 seconds (Warm-up)") != std::string::npos);
  EXPECT_TRUE(diag_output.find("20.3 seconds (Sampling)") != std::string::npos);
  EXPECT_TRUE(diag_output.find("30.8 seconds (Total)") != std::string::npos);
}

TEST_F(DispatcherFunctionsTest, GenerateTransitions) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int num_iterations = 3;  // Keep small for test
  int start = 0;
  int finish = 10;
  int num_thin = 1;
  int refresh = 0;
  bool save = true;
  bool warmup = true;

  stan::rng_t rng = stan::services::util::create_rng(seed, chain);

  std::vector<double> cont_vector = stan::services::util::initialize(
      model, context, rng, init_radius, false, logger, dispatcher);

  stan::mcmc::adapt_diag_e_nuts<stan_model, stan::rng_t> sampler(model, rng);

  // Initialize the sampler with reasonable defaults
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.set_max_depth(5);

  Eigen::VectorXd cont_params(cont_vector.size());
  for (size_t i = 0; i < cont_vector.size(); i++)
    cont_params[i] = cont_vector[i];
  stan::mcmc::sample s(cont_params, 0, 0);

  // First write the header to get num_model_values
  size_t num_model_values = 0;
  stan::services::util::write_sample_header(s, sampler, model, dispatcher,
                                            num_model_values);

  // Clear the streams
  sample_stream->str("");
  sample_stream->clear();
  diagnostic_stream->str("");
  diagnostic_stream->clear();

  // Run generate_transitions
  stan::services::util::generate_transitions(
      sampler, num_iterations, start, finish, num_thin, refresh, save, warmup,
      dispatcher, logger, num_model_values, s, model, rng, interrupt, chain, 1);

  // Check that we have output in the sample stream
  std::string sample_output = sample_stream->str();
  EXPECT_FALSE(sample_output.empty());

  // Count the number of lines in the sample output
  int line_count = 0;
  std::stringstream ss(sample_output);
  std::string line;
  while (std::getline(ss, line)) {
    if (!line.empty())
      line_count++;
  }

  // Should have at least num_iterations lines
  EXPECT_GE(line_count, num_iterations);
}

// A more comprehensive test that uses the hmc_nuts_diag_e_adapt sampler
TEST_F(DispatcherFunctionsTest, HMCNUTSDiagEAdapt) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int num_warmup = 5;   // Small for testing
  int num_samples = 5;  // Small for testing
  int num_thin = 1;
  bool save_warmup = false;
  int refresh = 0;
  double stepsize = 0.1;
  double stepsize_jitter = 0;
  int max_depth = 5;
  double delta = 0.8;
  double gamma = 0.05;
  double kappa = 0.75;
  double t0 = 10;
  unsigned int init_buffer = 2;
  unsigned int term_buffer = 2;
  unsigned int window = 2;

  // Run the NUTS sampler
  int result = stan::services::sample::hmc_nuts_diag_e_adapt(
      model, context, nullptr, seed, chain, init_radius, num_warmup,
      num_samples, num_thin, save_warmup, refresh, stepsize, stepsize_jitter,
      max_depth, delta, gamma, kappa, t0, init_buffer, term_buffer, window,
      &interrupt, &logger, &dispatcher);

  // Check that sampling completed successfully
  EXPECT_EQ(result, stan::services::error_codes::OK);

  // Check that we have output in the init stream
  std::string init_output = init_stream->str();
  EXPECT_FALSE(init_output.empty());

  // Check that we have output in the sample stream
  std::string sample_output = sample_stream->str();
  EXPECT_FALSE(sample_output.empty());

  // Check that we have output in the diagnostic stream
  std::string diag_output = diagnostic_stream->str();
  EXPECT_FALSE(diag_output.empty());

  // Check that we have metric output
  std::string metric_output = metric_stream->str();
  EXPECT_FALSE(metric_output.empty());

  // Count the number of lines in the sample output
  int line_count = 0;
  std::stringstream ss(sample_output);
  std::string line;
  while (std::getline(ss, line)) {
    if (!line.empty())
      line_count++;
  }

  // Should have at least num_warmup + num_samples + some header/info lines
  EXPECT_GE(line_count, num_warmup + num_samples);

  // Check for adaptation message
  EXPECT_TRUE(sample_output.find("Adaptation terminated") != std::string::npos);
}
