#include <stan/run/config_hmc_output.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>
  
// Create a deleter_noop struct for the json_writer
struct deleter_noop {
  template <typename T>
  constexpr void operator()(T* arg) const {}
};

class OutputConfigTest : public testing::Test {
protected:
  void SetUp() override {
    // Create writers
    init_writer = std::make_unique<stan::callbacks::stream_writer>(init_stream);
    sample_writer = std::make_unique<stan::callbacks::stream_writer>(sample_stream);
    diagnostic_writer = std::make_unique<stan::callbacks::stream_writer>(diagnostic_stream);
  }

  // Streams for output
  std::stringstream init_stream;
  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  
  // Writers (using unique_ptr for automatic cleanup)
  std::unique_ptr<stan::callbacks::stream_writer> init_writer;
  std::unique_ptr<stan::callbacks::stream_writer> sample_writer;
  std::unique_ptr<stan::callbacks::stream_writer> diagnostic_writer;
};

TEST_F(OutputConfigTest, DefaultConstructor) {
  // The default constructor should create an invalid configuration
  // that will throw when build() is called due to missing required components
  EXPECT_THROW(stan::run::config_hmc_output<>::create(1).build(),
               std::invalid_argument);
}

TEST_F(OutputConfigTest, SingleChainConstructor) {
  // Create a valid configuration with a single chain
  auto config_hmc_output = stan::run::config_hmc_output<>::create(1)
    .init_writer(init_writer.get())
    .sample_writer(sample_writer.get())
    .diagnostic_writer(diagnostic_writer.get())
    .build();
  
  // Verify the configuration
  EXPECT_EQ(init_writer.get(), config_hmc_output.init_writer());
  EXPECT_EQ(sample_writer.get(), config_hmc_output.sample_writer());
  EXPECT_EQ(diagnostic_writer.get(), config_hmc_output.diagnostic_writer());
  EXPECT_EQ(nullptr, config_hmc_output.metric_writer());
  EXPECT_FALSE(config_hmc_output.is_multichain());
  EXPECT_FALSE(config_hmc_output.has_metric_writer());
}



class OutputConfigMultiChainTest : public testing::Test {
protected:
  void SetUp() override {
    // Create 4 separate streams for each type of writer and each chain
    init_streams.resize(num_chains);
    sample_streams.resize(num_chains);
    diagnostic_streams.resize(num_chains);
    
    // Create writers for each chain
    for (size_t i = 0; i < num_chains; ++i) {
      init_writers.emplace_back(std::make_unique<stan::callbacks::stream_writer>(init_streams[i]));
      sample_writers.emplace_back(std::make_unique<stan::callbacks::stream_writer>(sample_streams[i]));
      diagnostic_writers.emplace_back(std::make_unique<stan::callbacks::stream_writer>(diagnostic_streams[i]));
    }
    
    // Prepare raw pointer vectors for the configuration
    init_raw_writers.resize(num_chains);
    sample_raw_writers.resize(num_chains);
    diagnostic_raw_writers.resize(num_chains);
    
    for (size_t i = 0; i < num_chains; ++i) {
      init_raw_writers[i] = init_writers[i].get();
      sample_raw_writers[i] = sample_writers[i].get();
      diagnostic_raw_writers[i] = diagnostic_writers[i].get();
    }
  }

  const size_t num_chains = 4;
  
  // Vectors of streams for each chain
  std::vector<std::stringstream> init_streams;
  std::vector<std::stringstream> sample_streams;
  std::vector<std::stringstream> diagnostic_streams;
  
  // Vectors of unique_ptr writers for ownership
  std::vector<std::unique_ptr<stan::callbacks::stream_writer>> init_writers;
  std::vector<std::unique_ptr<stan::callbacks::stream_writer>> sample_writers;
  std::vector<std::unique_ptr<stan::callbacks::stream_writer>> diagnostic_writers;
  
  // Vectors of raw pointers for the configuration
  std::vector<stan::callbacks::writer*> init_raw_writers;
  std::vector<stan::callbacks::writer*> sample_raw_writers;
  std::vector<stan::callbacks::writer*> diagnostic_raw_writers;
};

TEST_F(OutputConfigMultiChainTest, MultiChainConstructor) {
  // Create a valid configuration with multiple chains
  auto config_hmc_output = stan::run::config_hmc_output<>::create(num_chains)
    .init_writers(init_raw_writers)
    .sample_writers(sample_raw_writers)
    .diagnostic_writers(diagnostic_raw_writers)
    .build();
  
  // Verify the configuration
  EXPECT_TRUE(config_hmc_output.is_multichain());
  EXPECT_FALSE(config_hmc_output.has_metric_writer());
  
  // Verify that single-chain getters throw logic_error
  EXPECT_THROW(config_hmc_output.init_writer(), std::logic_error);
  EXPECT_THROW(config_hmc_output.sample_writer(), std::logic_error);
  EXPECT_THROW(config_hmc_output.diagnostic_writer(), std::logic_error);
  EXPECT_THROW(config_hmc_output.metric_writer(), std::logic_error);
}

// Test multi-chain getters return correct vectors
TEST_F(OutputConfigMultiChainTest, MultiChainGetters) {
  auto config_hmc_output = stan::run::config_hmc_output<>::create(num_chains)
    .init_writers(init_raw_writers)
    .sample_writers(sample_raw_writers)
    .diagnostic_writers(diagnostic_raw_writers)
    .build();
  
  // Check that the vectors returned by the getters match what we set
  EXPECT_EQ(init_raw_writers, config_hmc_output.init_writers());
  EXPECT_EQ(sample_raw_writers, config_hmc_output.sample_writers());
  EXPECT_EQ(diagnostic_raw_writers, config_hmc_output.diagnostic_writers());
}
