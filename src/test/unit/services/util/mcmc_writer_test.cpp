#include <iostream>
#include <stan/services/util/mcmc_writer.hpp>
#include <gtest/gtest.h>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/test-models/good/services/test_lp.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/services/util/create_rng.hpp>

namespace test {
// mock_throwing_model_in_write_array throws exception in the write_array()
// method
class throwing_model : public stan_model {
 public:

  throwing_model(stan::io::var_context &context, std::ostream* pstream) :
      stan_model(context, pstream) { }

  template <typename RNG>
  void write_array(RNG& base_rng__,
                   std::vector<double>& params_r__,
                   std::vector<int>& params_i__,
                   std::vector<double>& vars__,
                   bool include_tparams__ = true,
                   bool include_gqs__ = true,
                   std::ostream* pstream__ = 0) const {
    vars__.resize(0);
    for (size_t i = 0; i < params_r__.size() - 2; ++i)
      vars__.push_back(params_r__[i]);
    throw std::domain_error("throwing within write_array");
  }

  template <typename RNG>
  void write_array(RNG& base_rng,
                   Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                   Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                   bool include_tparams = true,
                   bool include_gqs = true,
                   std::ostream* pstream = 0) const {
    throw std::domain_error("throwing within write_array");
  }
};
}

class ServicesUtil : public ::testing::Test {
public:
  ServicesUtil()
    : mcmc_writer(sample_writer, diagnostic_writer, logger),
      model(context, &model_log),
      throwing_model(context, &model_log) {}

  stan::test::unit::instrumented_writer sample_writer, diagnostic_writer;
  stan::test::unit::instrumented_logger logger;
  stan::services::util::mcmc_writer mcmc_writer;
  std::stringstream model_log;
  stan::io::empty_var_context context;
  stan_model model;
  test::throwing_model throwing_model;
};

TEST_F(ServicesUtil, constructor) {
  EXPECT_EQ(0, sample_writer.call_count());
  EXPECT_EQ(0, diagnostic_writer.call_count());
  EXPECT_EQ(0, logger.call_count());
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
             stan::callbacks::logger& logger) {
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
  EXPECT_EQ(0, logger.call_count());

  EXPECT_EQ(2, mcmc_writer.num_sample_params_);
  EXPECT_EQ(0, mcmc_writer.num_sampler_params_);
  EXPECT_EQ(5, mcmc_writer.num_model_params_);
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
  EXPECT_EQ(0, logger.call_count());
}

TEST_F(ServicesUtil, write_adapt_finish) {
  mock_sampler sampler;

  mcmc_writer.write_adapt_finish(sampler);
  EXPECT_EQ(1, sample_writer.call_count());
  EXPECT_EQ(1, sample_writer.call_count("string"));
  EXPECT_EQ(0, diagnostic_writer.call_count());
  EXPECT_EQ(0, logger.call_count());
}

TEST_F(ServicesUtil, write_diagnostic_names) {
  Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
  stan::mcmc::sample sample(x, 1, 2);
  mock_sampler sampler;

  mcmc_writer.write_diagnostic_names(sample, sampler, model);
  EXPECT_EQ(0, sample_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count("vector_string"));
  EXPECT_EQ(0, logger.call_count());
}

TEST_F(ServicesUtil, write_diagnostic_params) {
  Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
  stan::mcmc::sample sample(x, 1, 2);
  mock_sampler sampler;

  mcmc_writer.write_diagnostic_params(sample, sampler);
  EXPECT_EQ(0, sample_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count("vector_double"));
  EXPECT_EQ(0, logger.call_count());
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
  EXPECT_EQ(5, logger.call_count());
  EXPECT_EQ(5, logger.call_count_info());
}


TEST_F(ServicesUtil, throwing_model__write_sample_parameters) {
  boost::ecuyer1988 rng = stan::services::util::create_rng(0, 1);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
  stan::mcmc::sample sample(x, 1, 2);
  mock_sampler sampler;

  mcmc_writer.write_sample_names(sample, sampler, throwing_model);
  ASSERT_EQ(5, mcmc_writer.num_model_params_);

  mcmc_writer.write_sample_params(rng, sample, sampler, throwing_model);
  EXPECT_EQ(2, sample_writer.call_count());
  EXPECT_EQ(1, sample_writer.call_count("vector_double"));
  EXPECT_EQ(0, diagnostic_writer.call_count());
  EXPECT_EQ(1, logger.call_count());


  std::vector<std::vector<double>> values = sample_writer.vector_double_values();
  ASSERT_EQ(1, values.size());
  ASSERT_EQ(mcmc_writer.num_sample_params_ + mcmc_writer.num_sampler_params_ + mcmc_writer.num_model_params_,
            values[0].size());

  for (size_t i = 0; i < mcmc_writer.num_sample_params_ + mcmc_writer.num_sampler_params_; ++i) {
    EXPECT_FALSE(std::isnan(values[0][i]));
  }

  for (size_t i = mcmc_writer.num_sample_params_ + mcmc_writer.num_sampler_params_;
       i < values[0].size(); ++i) {
    EXPECT_TRUE(std::isnan(values[0][i]));
  }
}
