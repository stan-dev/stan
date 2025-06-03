#include <stan/run/hmc_nuts.hpp>
#include <stan/run/hmc_nuts_config.hpp>
#include <stan/run/read_json_data.hpp>
#include <stan/run/load_model.hpp>
#include <stan/run/model_config.hpp>

#include <test/test-models/good/services/bernoulli.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>

#include <filesystem>
#include <memory>
#include <sstream>
#include <fstream>

#include <gtest/gtest.h>

// Helper function to count lines in a CSV file
size_t count_csv_lines(const std::filesystem::path& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    return 0;
  }
  
  size_t line_count = 0;
  std::string line;
  while (std::getline(file, line)) {
    line_count++;
  }
  return line_count;
}

// Helper function to count non-comment lines in Stan CSV output
size_t count_csv_data_lines(const std::filesystem::path& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    return 0;
  }
  
  size_t data_line_count = 0;
  std::string line;
  while (std::getline(file, line)) {
    // Skip comment lines (start with '#') and empty lines
    if (!line.empty() && line[0] != '#') {
      data_line_count++;
    }
  }
  return data_line_count;
}

// Helper function to find sample CSV file for a given chain
std::filesystem::path find_sample_csv(const std::filesystem::path& output_dir, 
                                      int chain_id = 1) {
  for (const auto& entry : std::filesystem::directory_iterator(output_dir)) {
    std::string filename = entry.path().filename().string();
    if (filename.find("sample") != std::string::npos && 
        filename.find("chain" + std::to_string(chain_id)) != std::string::npos &&
        entry.path().extension() == ".csv") {
      return entry.path();
    }
  }
  return std::filesystem::path{};  // Return empty path if not found
}

class HmcNutsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create temporary output directory
    temp_dir_ = std::filesystem::temp_directory_path() / "hmc_nuts_test";
    std::filesystem::create_directories(temp_dir_);
    
    // Create model configuration and load model using load_model function
    auto model_config = stan::run::model_config::create()
      .data("src/test/test-models/good/services/bernoulli.data.json")
      .seed(12345)
      .build();
    
    model_ = &stan::run::load_model(model_config);
  }
  
  void TearDown() override {
    std::filesystem::remove_all(temp_dir_);
  }
  
  std::filesystem::path temp_dir_;
  stan::model::model_base* model_;  // Pointer to model returned by load_model
};

TEST_F(HmcNutsTest, HmcNuts_SingleChain_Success) {
  const int num_warmup = 5;
  const int num_samples = 5;
  
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(num_warmup)
    .samples(num_samples)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)   // No progress messages
    .save_warmup(false)  // Don't save warmup samples
    .build();
  
  int result = stan::run::hmc_nuts(config, *model_);
  
  EXPECT_EQ(result, 0);  // Success
  
  // Check that sample file was created
  EXPECT_TRUE(std::filesystem::exists(temp_dir_));
  
  // Find and verify the sample CSV file
  auto sample_csv = find_sample_csv(temp_dir_, 1);
  EXPECT_FALSE(sample_csv.empty());
  
  if (!sample_csv.empty()) {
    size_t total_lines = count_csv_lines(sample_csv);
    size_t data_lines = count_csv_data_lines(sample_csv);
    
    // Data lines: header (1) + samples (num_samples) = 1 + 5 = 6 lines
    EXPECT_EQ(data_lines, 1 + num_samples);
    
    // Total lines should be much greater due to comment headers with metadata
    EXPECT_GT(total_lines, data_lines);
    EXPECT_GT(total_lines, 10);  // Should have substantial metadata comments
  }
}

TEST_F(HmcNutsTest, HmcNuts_MultipleChains_Success) {
  const int num_chains = 2;
  const int num_warmup = 3;
  const int num_samples = 3;
  
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(num_chains)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(num_warmup)
    .samples(num_samples)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .save_warmup(false)
    .build();
  
  int result = stan::run::hmc_nuts(config, *model_);
  
  EXPECT_EQ(result, 0);  // Success
  
  // Check that output files for multiple chains exist and have correct content
  for (int chain = 1; chain <= num_chains; ++chain) {
    auto sample_csv = find_sample_csv(temp_dir_, chain);
    EXPECT_FALSE(sample_csv.empty()) << "Sample CSV not found for chain " << chain;
    
    if (!sample_csv.empty()) {
      size_t total_lines = count_csv_lines(sample_csv);
      size_t data_lines = count_csv_data_lines(sample_csv);
      
      // Data lines: header (1) + samples (num_samples) = 1 + 3 = 4 lines
      EXPECT_EQ(data_lines, 1 + num_samples) << "Incorrect data line count for chain " << chain;
      
      // Total lines should include substantial metadata
      EXPECT_GT(total_lines, data_lines) << "Should have comment metadata for chain " << chain;
    }
  }
}

TEST_F(HmcNutsTest, HmcNuts_DifferentMetricTypes) {
  // Test DIAG_E metric
  auto config_diag = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::DIAG_E)
    .warmup(2)
    .samples(2)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .build();
  
  int result_diag = stan::run::hmc_nuts(config_diag, *model_);
  EXPECT_EQ(result_diag, 0);
  
  // Test DENSE_E metric
  auto config_dense = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::DENSE_E)
    .warmup(2)
    .samples(2)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .build();
  
  int result_dense = stan::run::hmc_nuts(config_dense, *model_);
  EXPECT_EQ(result_dense, 0);
}

TEST_F(HmcNutsTest, HmcNuts_MinimalSampling) {
  const int num_warmup = 1;
  const int num_samples = 1;
  
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(num_warmup)
    .samples(num_samples)
    .stepsize(1.0)
    .max_depth(3)
    .delta(0.8)
    .refresh(0)
    .save_warmup(false)
    .build();
  
  int result = stan::run::hmc_nuts(config, *model_);
  
  EXPECT_EQ(result, 0);
  
  // Verify CSV content even for minimal sampling
  auto sample_csv = find_sample_csv(temp_dir_, 1);
  EXPECT_FALSE(sample_csv.empty());
  
  if (!sample_csv.empty()) {
    size_t total_lines = count_csv_lines(sample_csv);
    size_t data_lines = count_csv_data_lines(sample_csv);
    
    // Data lines: header (1) + samples (1) = 2 lines
    EXPECT_EQ(data_lines, 1 + num_samples);
    
    // Should still have metadata comments even for minimal sampling
    EXPECT_GT(total_lines, data_lines);
  }
}

TEST_F(HmcNutsTest, HmcNuts_WithWarmupSaved) {
  const int num_warmup = 4;
  const int num_samples = 3;
  
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(num_warmup)
    .samples(num_samples)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .save_warmup(true)  // Save warmup samples
    .build();
  
  int result = stan::run::hmc_nuts(config, *model_);
  
  EXPECT_EQ(result, 0);
  
  // Verify CSV content includes warmup samples
  auto sample_csv = find_sample_csv(temp_dir_, 1);
  EXPECT_FALSE(sample_csv.empty());
  
  if (!sample_csv.empty()) {
    size_t total_lines = count_csv_lines(sample_csv);
    size_t data_lines = count_csv_data_lines(sample_csv);
    
    // Data lines: header (1) + warmup (4) + samples (3) = 8 lines
    EXPECT_EQ(data_lines, 1 + num_warmup + num_samples);
    
    // Total lines should include metadata comments
    EXPECT_GT(total_lines, data_lines);
  }
}

TEST_F(HmcNutsTest, HmcNuts_OutputFilesCreated) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(3)
    .samples(3)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .save_start_params(true)
    .save_diagnostics(true)
    .save_metric(true)
    .build();
  
  int result = stan::run::hmc_nuts(config, *model_);
  
  EXPECT_EQ(result, 0);
  
  // Check for different types of output files
  bool found_sample = false;
  bool found_start_params = false;
  bool found_diagnostics = false;
  bool found_metric = false;
  
  for (const auto& entry : std::filesystem::directory_iterator(temp_dir_)) {
    std::string filename = entry.path().filename().string();
    
    if (filename.find("sample") != std::string::npos && 
        entry.path().extension() == ".csv") {
      found_sample = true;
    }
    if (filename.find("start_params") != std::string::npos && 
        entry.path().extension() == ".csv") {
      found_start_params = true;
    }
    if (filename.find("param_grads") != std::string::npos && 
        entry.path().extension() == ".csv") {
      found_diagnostics = true;
    }
    if (filename.find("metric") != std::string::npos && 
        entry.path().extension() == ".json") {
      found_metric = true;
    }
  }
  
  EXPECT_TRUE(found_sample);
  EXPECT_TRUE(found_start_params);
  EXPECT_TRUE(found_diagnostics);
  EXPECT_TRUE(found_metric);
}

// Test with empty output directory (should use current directory)
TEST_F(HmcNutsTest, HmcNuts_EmptyOutputDir) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir("")  // Empty output directory
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(2)
    .samples(2)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .build();
  
  int result = stan::run::hmc_nuts(config, *model_);
  
  EXPECT_EQ(result, 0);
  
  // Clean up files created in current directory
  std::string model_name = model_->model_name();
  for (const auto& entry : std::filesystem::directory_iterator(".")) {
    std::string filename = entry.path().filename().string();
    if (filename.find(model_name) != std::string::npos && 
        entry.path().extension() == ".csv") {
      std::filesystem::remove(entry.path());
    }
    if (filename.find(model_name) != std::string::npos && 
        entry.path().extension() == ".json") {
      std::filesystem::remove(entry.path());
    }
  }
}

// Test error handling with invalid configuration
TEST_F(HmcNutsTest, HmcNuts_ErrorHandling) {
  // Test with extremely small stepsize that might cause issues
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(1)
    .samples(1)
    .stepsize(1e-10)  // Extremely small stepsize
    .max_depth(1)
    .delta(0.999)     // Very high delta
    .refresh(0)
    .build();
  
  // This might succeed or fail depending on the model, but shouldn't crash
  int result = stan::run::hmc_nuts(config, *model_);
  
  // Result should be either 0 (success) or 1 (error), not crash
  EXPECT_TRUE(result == 0 || result == 1);
}

// Test that function can be called multiple times
TEST_F(HmcNutsTest, HmcNuts_MultipleCalls) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(2)
    .samples(2)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .build();
  
  // First call
  int result1 = stan::run::hmc_nuts(config, *model_);
  EXPECT_EQ(result1, 0);
  
  // Second call with different seed
  auto config2 = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(54321)  // Different seed
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(2)
    .samples(2)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .build();
  
  int result2 = stan::run::hmc_nuts(config2, *model_);
  EXPECT_EQ(result2, 0);
}

// Test template parameter functionality
TEST_F(HmcNutsTest, HmcNuts_TemplateParameters) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(2)
    .samples(2)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .build();
  
  // Test with Jacobian = true (default)
  int result_true = stan::run::hmc_nuts<true>(config, *model_);
  EXPECT_EQ(result_true, 0);
  
  // Test with Jacobian = false
  int result_false = stan::run::hmc_nuts<false>(config, *model_);
  EXPECT_EQ(result_false, 0);
}

// Note: Testing fixed parameter models would require a model with no parameters,
// which is difficult to create with the current bernoulli model.
// This would be a good integration test for the future.

// Test that validates the chain count vs context count logic
TEST_F(HmcNutsTest, HmcNuts_ContextValidation) {
  // This test verifies that our fixed logic properly handles context counting
  const int num_chains = 3;
  const int num_warmup = 2;
  const int num_samples = 2;
  
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(num_chains)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(num_warmup)
    .samples(num_samples)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .save_warmup(false)
    .build();
  
  int result = stan::run::hmc_nuts(config, *model_);
  EXPECT_EQ(result, 0);
  
  // Verify CSV content for all chains
  for (int chain = 1; chain <= num_chains; ++chain) {
    auto sample_csv = find_sample_csv(temp_dir_, chain);
    EXPECT_FALSE(sample_csv.empty()) << "Sample CSV not found for chain " << chain;
    
    if (!sample_csv.empty()) {
      size_t total_lines = count_csv_lines(sample_csv);
      size_t data_lines = count_csv_data_lines(sample_csv);
      
      // Data lines: header (1) + samples (2) = 3 lines per chain
      EXPECT_EQ(data_lines, 1 + num_samples) << "Incorrect data line count for chain " << chain;
      
      // Should have metadata comments
      EXPECT_GT(total_lines, data_lines) << "Should have comment metadata for chain " << chain;
    }
  }
}
// Test with configuration that has save options disabled
TEST_F(HmcNutsTest, HmcNuts_MinimalOutput) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .output_dir(temp_dir_.string())
    .metric_type(stan::run::metric_t::UNIT_E)
    .warmup(2)
    .samples(2)
    .stepsize(1.0)
    .max_depth(5)
    .delta(0.8)
    .refresh(0)
    .save_start_params(false)  // Disable optional outputs
    .save_diagnostics(false)
    .save_metric(false)
    .build();
  
  int result = stan::run::hmc_nuts(config, *model_);
  
  EXPECT_EQ(result, 0);
  
  // Should still have sample files, but not the optional ones
  bool found_sample = false;
  bool found_optional = false;
  
  for (const auto& entry : std::filesystem::directory_iterator(temp_dir_)) {
    std::string filename = entry.path().filename().string();
    
    if (filename.find("sample") != std::string::npos && 
        entry.path().extension() == ".csv") {
      found_sample = true;
    }
    if ((filename.find("start_params") != std::string::npos ||
         filename.find("param_grads") != std::string::npos ||
         filename.find("metric") != std::string::npos)) {
      found_optional = true;
    }
  }
  
  EXPECT_TRUE(found_sample);
  EXPECT_FALSE(found_optional);  // Optional outputs should not exist
}
