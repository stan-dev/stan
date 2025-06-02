#include <stan/run/hmc_output_writers.hpp>
#include <stan/run/hmc_nuts_config.hpp>
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <string>

class HmcOutputWritersTest : public ::testing::Test {
protected:
  void SetUp() override {
    test_dir = std::filesystem::temp_directory_path() / "hmc_output_writers_test";
    std::filesystem::create_directories(test_dir);
  }
  
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);
  }
  
  std::filesystem::path test_dir;
};

TEST_F(HmcOutputWritersTest, CreateSingleChainWriters_AllOutputs) {
  auto config = stan::run::hmc_nuts_config::create()
    .output_dir(test_dir.string())
    .save_start_params(true)
    .save_diagnostics(true)
    .save_metric(true)
    .build();
  
  std::string timestamp = stan::run::generate_timestamp();
  auto writers = stan::run::create_hmc_nuts_single_chain_writers(
    config, "test_model", timestamp, 1, "# ");
  
  // All writers should be non-null
  EXPECT_NE(writers.sample_writer, nullptr);
  EXPECT_NE(writers.start_params_writer, nullptr);
  EXPECT_NE(writers.diagnostics_writer, nullptr);
  EXPECT_NE(writers.metric_writer, nullptr);
}

TEST_F(HmcOutputWritersTest, CreateSingleChainWriters_MinimalOutputs) {
  auto config = stan::run::hmc_nuts_config::create()
    .output_dir(test_dir.string())
    .save_start_params(false)
    .save_diagnostics(false)
    .save_metric(false)
    .build();
  
  std::string timestamp = stan::run::generate_timestamp();
  auto writers = stan::run::create_hmc_nuts_single_chain_writers(
    config, "test_model", timestamp, 1, "# ");
  
  // Only sample writer should be non-null
  EXPECT_NE(writers.sample_writer, nullptr);
  EXPECT_EQ(writers.start_params_writer, nullptr);
  EXPECT_EQ(writers.diagnostics_writer, nullptr);
  EXPECT_EQ(writers.metric_writer, nullptr);
}

TEST_F(HmcOutputWritersTest, CreateMultiChainWriters) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(3)
    .output_dir(test_dir.string())
    .save_start_params(true)
    .save_diagnostics(true)
    .save_metric(true)
    .build();
  
  auto writers_vec = stan::run::create_hmc_nuts_multi_chain_writers(
    config, "test_model", "# ");
  
  EXPECT_EQ(writers_vec.size(), 3);
  
  // Check that all chains have the expected writers
  for (size_t i = 0; i < writers_vec.size(); ++i) {
    EXPECT_NE(writers_vec[i].sample_writer, nullptr);
    EXPECT_NE(writers_vec[i].start_params_writer, nullptr);
    EXPECT_NE(writers_vec[i].diagnostics_writer, nullptr);
    EXPECT_NE(writers_vec[i].metric_writer, nullptr);
  }
}

TEST_F(HmcOutputWritersTest, CreateMultiChainWriters_SingleChain) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .output_dir(test_dir.string())
    .save_start_params(true)
    .save_diagnostics(false)
    .save_metric(true)
    .build();
  
  auto writers_vec = stan::run::create_hmc_nuts_multi_chain_writers(
    config, "test_model", "# ");
  
  EXPECT_EQ(writers_vec.size(), 1);
  EXPECT_NE(writers_vec[0].sample_writer, nullptr);
  EXPECT_NE(writers_vec[0].start_params_writer, nullptr);
  EXPECT_EQ(writers_vec[0].diagnostics_writer, nullptr);
  EXPECT_NE(writers_vec[0].metric_writer, nullptr);
}

TEST_F(HmcOutputWritersTest, WritersActuallyCreateFiles) {
  auto config = stan::run::hmc_nuts_config::create()
    .output_dir(test_dir.string())
    .save_start_params(true)
    .save_diagnostics(true)
    .save_metric(true)
    .build();
  
  std::string timestamp = stan::run::generate_timestamp();
  auto writers = stan::run::create_hmc_nuts_single_chain_writers(
    config, "test_model", timestamp, 1, "# ");
  
  // Write some test data
  std::vector<std::string> headers = {"param1", "param2"};
  std::vector<double> values = {1.5, 2.5};
  
  writers.sample_writer->operator()(headers);
  writers.sample_writer->operator()(values);
  
  writers.start_params_writer->operator()(headers);
  writers.start_params_writer->operator()(values);
  
  writers.diagnostics_writer->operator()(headers);
  writers.diagnostics_writer->operator()(values);
  
  writers.metric_writer->begin_record();
  writers.metric_writer->write("test");
  writers.metric_writer->end_record();
  
  // Reset writers to flush files
  writers.sample_writer.reset();
  writers.start_params_writer.reset();
  writers.diagnostics_writer.reset();
  writers.metric_writer.reset();
  
  // Check that files were created
  std::string sample_file = stan::run::create_file_path(
    test_dir.string(),
    stan::run::generate_filename("test_model", timestamp, 1, "sample", ".csv")
  );
  
  std::string start_params_file = stan::run::create_file_path(
    test_dir.string(),
    stan::run::generate_filename("test_model", timestamp, 1, "start_params", ".csv")
  );
  
  std::string diagnostics_file = stan::run::create_file_path(
    test_dir.string(),
    stan::run::generate_filename("test_model", timestamp, 1, "param_grads", ".csv")
  );
  
  std::string metric_file = stan::run::create_file_path(
    test_dir.string(),
    stan::run::generate_filename("test_model", timestamp, 1, "metric", ".json")
  );
  
  EXPECT_TRUE(std::filesystem::exists(sample_file));
  EXPECT_TRUE(std::filesystem::exists(start_params_file));
  EXPECT_TRUE(std::filesystem::exists(diagnostics_file));
  EXPECT_TRUE(std::filesystem::exists(metric_file));
}
