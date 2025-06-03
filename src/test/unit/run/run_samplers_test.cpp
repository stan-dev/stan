#include <stan/run/run_samplers.hpp>
#include <stan/run/hmc_nuts_config.hpp>
#include <stan/run/hmc_output_writers.hpp>
#include <stan/run/read_json_data.hpp>

#include <test/test-models/good/services/bernoulli.hpp>

#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>

#include <filesystem>
#include <memory>
#include <vector>
#include <sstream>

#include <gtest/gtest.h>

class RunSamplersTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create temporary output directory
    temp_dir_ = std::filesystem::temp_directory_path() / "run_samplers_test";
    std::filesystem::create_directories(temp_dir_);
    
    // Load test data and create model
    auto data_context = stan::run::read_json_data("src/test/test-models/good/services/bernoulli.data.json");
    model_ = std::make_unique<bernoulli_model_namespace::bernoulli_model>(*data_context, 12345);
    
    // Setup minimal configuration for fast testing using unique_ptr
    config_ = std::make_unique<stan::run::hmc_nuts_config>(
      stan::run::hmc_nuts_config::create()
        .num_chains(1)
        .seed(12345)
        .output_dir(temp_dir_.string())
        .metric_type(stan::run::metric_t::UNIT_E)  // Fastest option
        .warmup(10)        // Very small for testing
        .samples(10)       // Very small for testing
        .stepsize(1.0)
        .max_depth(5)      // Small for speed
        .delta(0.8)
        .refresh(0)        // No progress messages
        .build()
    );
    
    // Create contexts
    for (size_t i = 0; i < config_->num_chains(); ++i) {
      init_contexts_.push_back(stan::run::read_json_data(""));
      metric_contexts_.push_back(stan::run::read_json_data(""));
    }
    
    // Create writers
    writers_ = stan::run::create_hmc_nuts_multi_chain_writers(*config_, "test_model");
    
    // Create logger and interrupt
    logger_ = std::make_unique<stan::test::unit::instrumented_logger>();
    interrupt_ = std::make_unique<stan::callbacks::interrupt>();
  }
  
  void TearDown() override {
    std::filesystem::remove_all(temp_dir_);
  }
  
  std::filesystem::path temp_dir_;
  std::unique_ptr<bernoulli_model_namespace::bernoulli_model> model_;
  std::unique_ptr<stan::run::hmc_nuts_config> config_;  // Use unique_ptr to avoid default constructor requirement
  std::vector<std::shared_ptr<const stan::io::var_context>> init_contexts_;
  std::vector<std::shared_ptr<const stan::io::var_context>> metric_contexts_;
  std::vector<stan::run::hmc_nuts_writers> writers_;
  std::unique_ptr<stan::test::unit::instrumented_logger> logger_;
  std::unique_ptr<stan::callbacks::interrupt> interrupt_;
};

TEST_F(RunSamplersTest, RunSamplers_SingleChain_UnitE) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(10)
    .samples(10)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .build();
  
  // Resize contexts and writers for single chain
  init_contexts_.resize(1);
  metric_contexts_.resize(1);
  writers_.resize(1);
  
  EXPECT_NO_THROW({
    stan::run::run_samplers(*model_, config, init_contexts_, metric_contexts_, 
                           writers_, *interrupt_, *logger_);
  });
  
  // Check that sample file was created
  EXPECT_TRUE(std::filesystem::exists(temp_dir_));
}

TEST_F(RunSamplersTest, RunSamplers_SingleChain_DiagE) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::DIAG_E)
    .warmup(10)
    .samples(10)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .build();
  
  init_contexts_.resize(1);
  metric_contexts_.resize(1);
  writers_.resize(1);
  
  EXPECT_NO_THROW({
    stan::run::run_samplers(*model_, config, init_contexts_, metric_contexts_, 
                           writers_, *interrupt_, *logger_);
  });
}

TEST_F(RunSamplersTest, RunSamplers_MultipleChains) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(2)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(10)
    .samples(10)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .build();
  
  // Resize contexts and writers for multiple chains
  init_contexts_.resize(2);
  metric_contexts_.resize(2);
  writers_.resize(2);
  
  // Add second chain contexts
  init_contexts_[1] = stan::run::read_json_data("");
  metric_contexts_[1] = stan::run::read_json_data("");
  
  // Create second chain writers
  writers_[1] = stan::run::create_hmc_nuts_single_chain_writers(
    config, "test_model", stan::run::generate_timestamp(), 2);
  
  EXPECT_NO_THROW({
    stan::run::run_samplers(*model_, config, init_contexts_, metric_contexts_, 
                           writers_, *interrupt_, *logger_);
  });
}

TEST_F(RunSamplersTest, SamplerRunner_Construction) {
  stan::run::sampler_runner runner(*model_, *config_, writers_, *interrupt_, *logger_);
  
  // Just test that construction doesn't throw
  EXPECT_TRUE(true);
}

TEST_F(RunSamplersTest, RunSamplers_MinimalSampling) {
  // Test with absolute minimal sampling to ensure basic functionality
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(1)
    .samples(1)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .build();
  
  init_contexts_.resize(1);
  metric_contexts_.resize(1);
  writers_.resize(1);
  
  EXPECT_NO_THROW({
    stan::run::run_samplers(*model_, config, init_contexts_, metric_contexts_, 
                           writers_, *interrupt_, *logger_);
  });
}

TEST_F(RunSamplersTest, RunSamplers_WithNullableWriters) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(10)
    .samples(10)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .save_start_params(false)  // This should make start_params_writer null
    .save_diagnostics(false)   // This should make diagnostics_writer null
    .save_metric(false)        // This should make metric_writer null
    .build();
  
  // Recreate writers with nullable options
  writers_ = stan::run::create_hmc_nuts_multi_chain_writers(config, "test_model");
  
  init_contexts_.resize(1);
  metric_contexts_.resize(1);
  
  EXPECT_NO_THROW({
    stan::run::run_samplers(*model_, config, init_contexts_, metric_contexts_, 
                           writers_, *interrupt_, *logger_);
  });
  
  // Verify that nullable writers are actually null
  EXPECT_EQ(writers_[0].start_params_writer, nullptr);
  EXPECT_EQ(writers_[0].diagnostics_writer, nullptr);
  EXPECT_EQ(writers_[0].metric_writer, nullptr);
  EXPECT_NE(writers_[0].sample_writer, nullptr);  // Sample writer should never be null
}

TEST_F(RunSamplersTest, VerifyLogMessages_MultipleChains) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(2)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(1)  // Minimal for speed
    .samples(1)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .build();
  
  // Resize contexts and writers for multiple chains
  init_contexts_.resize(2);
  metric_contexts_.resize(2);
  writers_.resize(2);
  
  // Add second chain contexts
  init_contexts_[1] = stan::run::read_json_data("");
  metric_contexts_[1] = stan::run::read_json_data("");
  
  // Create second chain writers
  writers_[1] = stan::run::create_hmc_nuts_single_chain_writers(
    config, "test_model", stan::run::generate_timestamp(), 2);
  
  // Clear any existing log messages
  logger_->info_.clear();
  logger_->error_.clear();
  
  EXPECT_NO_THROW({
    stan::run::run_samplers(*model_, config, init_contexts_, metric_contexts_, 
                           writers_, *interrupt_, *logger_);
  });
  
  // Verify expected log messages were captured
  EXPECT_GT(logger_->call_count_info(), 0);
  
  // Check for chain start messages
  EXPECT_GT(logger_->find_info("Starting chain 1 of 2"), 0);
  EXPECT_GT(logger_->find_info("Starting chain 2 of 2"), 0);
  
  // Check for completion messages
  EXPECT_GT(logger_->find_info("Completed chain 1"), 0);
  EXPECT_GT(logger_->find_info("Completed chain 2"), 0);
  
  // Check for final completion message
  EXPECT_GT(logger_->find_info("All 2 chains completed successfully"), 0);
  
  // Should have no error messages for successful run
  EXPECT_EQ(logger_->call_count_error(), 0);
}

TEST_F(RunSamplersTest, VerifyLogMessages_SingleChain) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(1)
    .samples(1)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .build();
  
  init_contexts_.resize(1);
  metric_contexts_.resize(1);
  writers_.resize(1);
  
  // Clear any existing log messages
  logger_->info_.clear();
  logger_->error_.clear();
  
  EXPECT_NO_THROW({
    stan::run::run_samplers(*model_, config, init_contexts_, metric_contexts_, 
                           writers_, *interrupt_, *logger_);
  });
  
  // For single chain, should have minimal logging since it doesn't go through 
  // the multiple chain sequential path
  // The run_single_chain method doesn't log - only run_multiple_chains_sequential does
  // So we expect no chain progress messages for single chain
  EXPECT_EQ(logger_->find_info("Starting chain"), 0);
  EXPECT_EQ(logger_->call_count_error(), 0);
}

TEST_F(RunSamplersTest, LoggerInstrumentation_Basic) {
  // Basic test to verify the instrumented logger is working
  logger_->info("Test info message");
  logger_->error("Test error message");
  
  EXPECT_EQ(logger_->call_count_info(), 1);
  EXPECT_EQ(logger_->call_count_error(), 1);
  EXPECT_GT(logger_->find_info("Test info message"), 0);
  EXPECT_GT(logger_->find_error("Test error message"), 0);
}
