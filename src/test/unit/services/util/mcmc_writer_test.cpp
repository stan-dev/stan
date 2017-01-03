#include <stan/services/util/mcmc_writer.hpp>
#include <gtest/gtest.h>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/test-models/good/services/test_lp.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/services/util/create_rng.hpp>

class ServicesUtil : public ::testing::Test {
public:
  ServicesUtil()
    : mcmc_writer(sample_writer, diagnostic_writer, message_writer),
      model(context, &model_log) { }

  stan::test::unit::instrumented_writer sample_writer, diagnostic_writer, message_writer;
  stan::services::util::mcmc_writer mcmc_writer;
  std::stringstream model_log;
  stan::io::empty_var_context context;
  stan_model model;
};

TEST_F(ServicesUtil, constructor) {
  EXPECT_EQ(0, sample_writer.call_count());
  EXPECT_EQ(0, diagnostic_writer.call_count());
  EXPECT_EQ(0, message_writer.call_count());
}

class mock_sampler : public stan::mcmc::base_mcmc {
public:
  int n_transition;
  int n_get_sampler_param_names;
  int n_get_sampler_params;
  int n_write_sampler_state;
  int n_get_sampler_diagnostic_names;
  int n_get_sampler_diagnostics;

  mock_sampler() {
    reset();
  }

  void reset() {
    n_transition = 0;
    n_get_sampler_param_names = 0;
    n_get_sampler_params = 0;
    n_write_sampler_state = 0;
    n_get_sampler_diagnostic_names = 0;
    n_get_sampler_diagnostics = 0;
  }

  stan::mcmc::sample
  transition(stan::mcmc::sample& init_sample,
             stan::callbacks::writer& info_writer,
             stan::callbacks::writer& error_writer) {
    ++n_transition;
    stan::mcmc::sample result(init_sample);
    return result;
  }

  void get_sampler_param_names(std::vector<std::string>& names) {
    ++n_get_sampler_param_names;
  }

  void get_sampler_params(std::vector<double>& values) {
    ++n_get_sampler_params;
  }

  void write_sampler_state(stan::callbacks::writer& writer) {
    ++n_write_sampler_state;
  }

  void get_sampler_diagnostic_names(std::vector<std::string>& model_names,
                                    std::vector<std::string>& names) {
    ++n_get_sampler_diagnostic_names;
  }

  void get_sampler_diagnostics(std::vector<double>& values) {
    ++n_get_sampler_diagnostics;
  }
};

TEST_F(ServicesUtil, write_sample_names) {
  Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
  stan::mcmc::sample sample(x, 1, 2);
  mock_sampler sampler;

  mcmc_writer.write_sample_names(sample, sampler, model);
  EXPECT_EQ(1, sample_writer.call_count());
  EXPECT_EQ(1, sample_writer.call_count("vector_string"));
  EXPECT_EQ(0, diagnostic_writer.call_count());
  EXPECT_EQ(0, message_writer.call_count());
}

TEST_F(ServicesUtil, write_sample_params) {
  boost::ecuyer1988 rng = stan::services::util::create_rng(0, 1);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
  stan::mcmc::sample sample(x, 1, 2);
  mock_sampler sampler;

  mcmc_writer.write_sample_params(rng, sample, sampler, model);
  EXPECT_EQ(1, sample_writer.call_count());
  EXPECT_EQ(1, sample_writer.call_count("vector_double"));
  EXPECT_EQ(0, diagnostic_writer.call_count());
  EXPECT_EQ(0, message_writer.call_count());
}

TEST_F(ServicesUtil, write_adapt_finish) {
  mock_sampler sampler;

  mcmc_writer.write_adapt_finish(sampler);
  EXPECT_EQ(1, sample_writer.call_count());
  EXPECT_EQ(1, sample_writer.call_count("string"));
  EXPECT_EQ(0, diagnostic_writer.call_count());
  EXPECT_EQ(0, message_writer.call_count());
}

TEST_F(ServicesUtil, write_diagnostic_names) {
  Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
  stan::mcmc::sample sample(x, 1, 2);
  mock_sampler sampler;

  mcmc_writer.write_diagnostic_names(sample, sampler, model);
  EXPECT_EQ(0, sample_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count("vector_string"));
  EXPECT_EQ(0, message_writer.call_count());
}

TEST_F(ServicesUtil, write_diagnostic_params) {
  Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
  stan::mcmc::sample sample(x, 1, 2);
  mock_sampler sampler;

  mcmc_writer.write_diagnostic_params(sample, sampler);
  EXPECT_EQ(0, sample_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count("vector_double"));
  EXPECT_EQ(0, message_writer.call_count());
}

TEST_F(ServicesUtil, internal_write_timing) {
  stan::test::unit::instrumented_writer writer;
  mcmc_writer.write_timing(0, 0, writer);
  EXPECT_EQ(5, writer.call_count());
  EXPECT_EQ(3, writer.call_count("string"));
  EXPECT_EQ(2, writer.call_count("empty"));
}

TEST_F(ServicesUtil, write_timing) {
  mcmc_writer.write_timing(0, 0);
  EXPECT_EQ(5, sample_writer.call_count());
  EXPECT_EQ(3, sample_writer.call_count("string"));
  EXPECT_EQ(2, sample_writer.call_count("empty"));
  EXPECT_EQ(5, diagnostic_writer.call_count());
  EXPECT_EQ(3, diagnostic_writer.call_count("string"));
  EXPECT_EQ(2, diagnostic_writer.call_count("empty"));
  EXPECT_EQ(5, message_writer.call_count());
  EXPECT_EQ(3, message_writer.call_count("string"));
  EXPECT_EQ(2, message_writer.call_count("empty"));
}
