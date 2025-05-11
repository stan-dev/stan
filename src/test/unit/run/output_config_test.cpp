#include <stan/run/config_output.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
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
    // Create the logger that writes to all streams
    logger = std::make_unique<stan::callbacks::stream_logger>(
        logger_stream, logger_stream, logger_stream, logger_stream, logger_stream);
    
    // Create writers
    init_writer = std::make_unique<stan::callbacks::stream_writer>(init_stream);
    sample_writer = std::make_unique<stan::callbacks::stream_writer>(sample_stream);
    diagnostic_writer = std::make_unique<stan::callbacks::stream_writer>(diagnostic_stream);
  }

  // Streams for output
  std::stringstream logger_stream;
  std::stringstream init_stream;
  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  
  // Writers and logger (using unique_ptr for automatic cleanup)
  std::unique_ptr<stan::callbacks::stream_logger> logger;
  std::unique_ptr<stan::callbacks::stream_writer> init_writer;
  std::unique_ptr<stan::callbacks::stream_writer> sample_writer;
  std::unique_ptr<stan::callbacks::stream_writer> diagnostic_writer;
};

TEST_F(OutputConfigTest, DefaultConstructor) {
  // The default constructor should create an invalid configuration
  // that will throw when build() is called due to missing required components
  EXPECT_THROW(stan::run::config_output<>::create().build(),
               std::invalid_argument);
}

TEST_F(OutputConfigTest, SingleChainConstructor) {
  // Create a valid configuration with a single chain
  auto config_output = stan::run::config_output<>::create()
    .logger(logger.get())
    .init_writer(init_writer.get())
    .sample_writer(sample_writer.get())
    .diagnostic_writer(diagnostic_writer.get())
    .build();
  
  // Verify the configuration
  EXPECT_EQ(1, config_output.num_chains());
  EXPECT_EQ(logger.get(), config_output.logger());
  EXPECT_EQ(init_writer.get(), config_output.init_writer());
  EXPECT_EQ(sample_writer.get(), config_output.sample_writer());
  EXPECT_EQ(diagnostic_writer.get(), config_output.diagnostic_writer());
  EXPECT_EQ(nullptr, config_output.metric_writer());
  EXPECT_FALSE(config_output.is_multichain());
  EXPECT_FALSE(config_output.has_metric_writer());
}



class OutputConfigMultiChainTest : public testing::Test {
protected:
  void SetUp() override {
    // Create the logger that writes to all streams
    logger = std::make_unique<stan::callbacks::stream_logger>(
        logger_stream, logger_stream, logger_stream, logger_stream, logger_stream);
    
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
  
  // Single stream for logger
  std::stringstream logger_stream;
  
  // Vectors of streams for each chain
  std::vector<std::stringstream> init_streams;
  std::vector<std::stringstream> sample_streams;
  std::vector<std::stringstream> diagnostic_streams;
  
  // Logger (using unique_ptr for automatic cleanup)
  std::unique_ptr<stan::callbacks::stream_logger> logger;
  
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
  auto config_output = stan::run::config_output<>::create()
    .num_chains(num_chains)
    .logger(logger.get())
    .init_writers(init_raw_writers)
    .sample_writers(sample_raw_writers)
    .diagnostic_writers(diagnostic_raw_writers)
    .build();
  
  // Verify the configuration
  EXPECT_EQ(num_chains, config_output.num_chains());
  EXPECT_EQ(logger.get(), config_output.logger());
  EXPECT_TRUE(config_output.is_multichain());
  EXPECT_FALSE(config_output.has_metric_writer());
  
  // Verify individual chain writers
  for (size_t i = 0; i < num_chains; ++i) {
    EXPECT_EQ(init_raw_writers[i], config_output.init_writer(i));
    EXPECT_EQ(sample_raw_writers[i], config_output.sample_writer(i));
    EXPECT_EQ(diagnostic_raw_writers[i], config_output.diagnostic_writer(i));
    EXPECT_EQ(nullptr, config_output.metric_writer(i));
  }
  
  // Verify that single-chain getters throw logic_error
  EXPECT_THROW(config_output.init_writer(), std::logic_error);
  EXPECT_THROW(config_output.sample_writer(), std::logic_error);
  EXPECT_THROW(config_output.diagnostic_writer(), std::logic_error);
  EXPECT_THROW(config_output.metric_writer(), std::logic_error);
}

// Test that individual missing writers cause validation to fail
TEST_F(OutputConfigTest, MissingIndividualWriters) {
  // Missing init_writer
  EXPECT_THROW(
    stan::run::config_output<>::create()
      .logger(logger.get())
      .sample_writer(sample_writer.get())
      .diagnostic_writer(diagnostic_writer.get())
      .build(),
    std::invalid_argument
  );
  
  // Missing sample_writer
  EXPECT_THROW(
    stan::run::config_output<>::create()
      .logger(logger.get())
      .init_writer(init_writer.get())
      .diagnostic_writer(diagnostic_writer.get())
      .build(),
    std::invalid_argument
  );
  
  // Missing diagnostic_writer
  EXPECT_THROW(
    stan::run::config_output<>::create()
      .logger(logger.get())
      .init_writer(init_writer.get())
      .sample_writer(sample_writer.get())
      .build(),
    std::invalid_argument
  );
  
  // Missing logger
  EXPECT_THROW(
    stan::run::config_output<>::create()
      .init_writer(init_writer.get())
      .sample_writer(sample_writer.get())
      .diagnostic_writer(diagnostic_writer.get())
      .build(),
    std::invalid_argument
  );
}

// Test that out-of-bounds chain indices throw exceptions
TEST_F(OutputConfigMultiChainTest, InvalidChainIndex) {
  auto config_output = stan::run::config_output<>::create()
    .num_chains(num_chains)
    .logger(logger.get())
    .init_writers(init_raw_writers)
    .sample_writers(sample_raw_writers)
    .diagnostic_writers(diagnostic_raw_writers)
    .build();
  
  // Using an out-of-bounds index should throw
  size_t invalid_index = num_chains;
  EXPECT_THROW(config_output.init_writer(invalid_index), std::out_of_range);
  EXPECT_THROW(config_output.sample_writer(invalid_index), std::out_of_range);
  EXPECT_THROW(config_output.diagnostic_writer(invalid_index), std::out_of_range);
  EXPECT_THROW(config_output.metric_writer(invalid_index), std::out_of_range);
}

// Test setting writers for individual chains
TEST_F(OutputConfigMultiChainTest, IndividualChainSetters) {
  auto builder = stan::run::config_output<>::create()
    .num_chains(num_chains)
    .logger(logger.get());
  
  // Set writers for each chain individually
  for (size_t i = 0; i < num_chains; ++i) {
    builder.init_writer(i, init_raw_writers[i])
           .sample_writer(i, sample_raw_writers[i])
           .diagnostic_writer(i, diagnostic_raw_writers[i]);
  }
  
  auto config_output = builder.build();
  
  // Verify each chain's writers
  for (size_t i = 0; i < num_chains; ++i) {
    EXPECT_EQ(init_raw_writers[i], config_output.init_writer(i));
    EXPECT_EQ(sample_raw_writers[i], config_output.sample_writer(i));
    EXPECT_EQ(diagnostic_raw_writers[i], config_output.diagnostic_writer(i));
  }
}

// Test vector size validation
TEST_F(OutputConfigMultiChainTest, VectorSizeValidation) {
  auto builder = stan::run::config_output<>::create()
    .num_chains(num_chains)
    .logger(logger.get());
  
  // Create undersized vectors
  std::vector<stan::callbacks::writer*> undersized = {init_raw_writers[0], init_raw_writers[1]};
  ASSERT_LT(undersized.size(), num_chains);
  
  // Creating with an undersized vector should throw
  EXPECT_THROW(builder.init_writers(undersized), std::invalid_argument);
  EXPECT_THROW(builder.sample_writers(undersized), std::invalid_argument);
  EXPECT_THROW(builder.diagnostic_writers(undersized), std::invalid_argument);
  
  // Create oversized vectors
  std::vector<stan::callbacks::writer*> oversized = init_raw_writers;
  oversized.push_back(init_raw_writers[0]);
  ASSERT_GT(oversized.size(), num_chains);
  
  // Creating with an oversized vector should throw
  EXPECT_THROW(builder.init_writers(oversized), std::invalid_argument);
  EXPECT_THROW(builder.sample_writers(oversized), std::invalid_argument);
  EXPECT_THROW(builder.diagnostic_writers(oversized), std::invalid_argument);
}

// Test multi-chain getters return correct vectors
TEST_F(OutputConfigMultiChainTest, MultiChainGetters) {
  auto config_output = stan::run::config_output<>::create()
    .num_chains(num_chains)
    .logger(logger.get())
    .init_writers(init_raw_writers)
    .sample_writers(sample_raw_writers)
    .diagnostic_writers(diagnostic_raw_writers)
    .build();
  
  // Check that the vectors returned by the getters match what we set
  EXPECT_EQ(init_raw_writers, config_output.init_writers());
  EXPECT_EQ(sample_raw_writers, config_output.sample_writers());
  EXPECT_EQ(diagnostic_raw_writers, config_output.diagnostic_writers());
}

// Test optional metric writer
TEST_F(OutputConfigTest, OptionalMetricWriter) {
  // Add metric_stream to the test fixture
  std::stringstream metric_stream;
  
  // Create a metric writer as json_writer using the noop deleter pattern
  auto metric_writer = std::make_unique<stan::callbacks::json_writer<std::stringstream, deleter_noop>>(
      std::unique_ptr<std::stringstream, deleter_noop>(&metric_stream));
  
  // Test with metric writer
  auto config_with_metric = stan::run::config_output<stan::callbacks::writer, stan::callbacks::structured_writer>::create()
    .logger(logger.get())
    .init_writer(init_writer.get())
    .sample_writer(sample_writer.get())
    .diagnostic_writer(diagnostic_writer.get())
    .metric_writer(metric_writer.get())
    .build();
  
  EXPECT_TRUE(config_with_metric.has_metric_writer());
  EXPECT_EQ(metric_writer.get(), config_with_metric.metric_writer());
  
  // Test without metric writer
  auto config_without_metric = stan::run::config_output<>::create()
    .logger(logger.get())
    .init_writer(init_writer.get())
    .sample_writer(sample_writer.get())
    .diagnostic_writer(diagnostic_writer.get())
    .build();
  
  EXPECT_FALSE(config_without_metric.has_metric_writer());
  EXPECT_EQ(nullptr, config_without_metric.metric_writer());
}
