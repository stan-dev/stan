#include <stan/run/output_config.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <functional>

class OutputConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up a logger for testing
    logger_stream = std::make_shared<std::stringstream>();
    logger = std::make_shared<TestLogger>(logger_stream);
  }

  /**
   * Custom deleter that doesn't delete the pointer.
   */
  struct deleter_noop {
    template <typename T>
    void operator()(T* ptr) const {}
  };

  class TestLogger : public stan::callbacks::logger {
   public:
    explicit TestLogger(std::shared_ptr<std::stringstream> stream)
        : stream_(stream) {}

    void info(const std::string& message) override { *stream_ << message << std::endl; }
    void error(const std::string& message) override { *stream_ << "ERROR: " << message << std::endl; }
    void warn(const std::string& message) override { *stream_ << "WARNING: " << message << std::endl; }
    void debug(const std::string& message) override { *stream_ << "DEBUG: " << message << std::endl; }
    void fatal(const std::string& message) override { *stream_ << "FATAL: " << message << std::endl; }

    void info(const std::stringstream& message) override { *stream_ << message.str() << std::endl; }
    void error(const std::stringstream& message) override { *stream_ << "ERROR: " << message.str() << std::endl; }
    void warn(const std::stringstream& message) override { *stream_ << "WARNING: " << message.str() << std::endl; }
    void debug(const std::stringstream& message) override { *stream_ << "DEBUG: " << message.str() << std::endl; }
    void fatal(const std::stringstream& message) override { *stream_ << "FATAL: " << message.str() << std::endl; }

    std::shared_ptr<std::stringstream> stream_;
  };

  std::shared_ptr<std::stringstream> logger_stream;
  std::shared_ptr<TestLogger> logger;
  
  // Create unique_stream_writer with noop deleter
  std::unique_ptr<stan::callbacks::unique_stream_writer<std::stringstream, deleter_noop>> 
  create_unique_writer() {
    auto stream = std::make_unique<std::stringstream>();
    std::stringstream* stream_ptr = stream.get();
    
    auto writer = std::make_unique<stan::callbacks::unique_stream_writer<std::stringstream, deleter_noop>>(
        std::unique_ptr<std::stringstream, deleter_noop>(stream_ptr));
    
    // Don't delete the stream when test ends
    stream.release();
    
    return writer;
  }
  
  // Create json_writer with noop deleter
  std::unique_ptr<stan::callbacks::json_writer<std::stringstream, deleter_noop>> 
  create_json_writer() {
    auto stream = std::make_unique<std::stringstream>();
    std::stringstream* stream_ptr = stream.get();
    
    auto writer = std::make_unique<stan::callbacks::json_writer<std::stringstream, deleter_noop>>(
        std::unique_ptr<std::stringstream, deleter_noop>(stream_ptr));
    
    // Don't delete the stream when test ends
    stream.release();
    
    return writer;
  }
  
  // Helper to check out of range access
  template <typename Func>
  void expect_out_of_range(Func func) {
    try {
      func();
      FAIL() << "Expected std::out_of_range exception";
    } catch(const std::out_of_range&) {
      // Expected
    } catch(...) {
      FAIL() << "Expected std::out_of_range exception, got different exception";
    }
  }
};

TEST_F(OutputConfigTest, DefaultConstructor) {
  using stan::run::output_config;
  
  // Create config with default constructor
  output_config<std::stringstream, deleter_noop> config;

  // Verify default values for single chain
  EXPECT_EQ(1, config.num_chains());
  EXPECT_FALSE(config.is_multichain());
  
  // All pointers should be null
  EXPECT_EQ(nullptr, config.logger());
  EXPECT_EQ(nullptr, config.init_writer());
  EXPECT_EQ(nullptr, config.sample_writer());
  EXPECT_EQ(nullptr, config.diagnostic_writer());
  EXPECT_EQ(nullptr, config.metric_writer());
  
  // Config should not be valid
  EXPECT_FALSE(config.is_valid());
  EXPECT_FALSE(config.has_metric_writer());
}

TEST_F(OutputConfigTest, SingleChainConstructor) {
  using config_t = stan::run::output_config<std::stringstream, deleter_noop>;
  using writer_t = stan::callbacks::unique_stream_writer<std::stringstream, deleter_noop>;
  
  // Create test objects
  auto init_writer = create_unique_writer();
  auto sample_writer = create_unique_writer();
  auto diagnostic_writer = create_unique_writer();
  
  // Create config with only required writers
  config_t config(logger.get(), init_writer.get(), 
                 sample_writer.get(), diagnostic_writer.get());

  // Verify configuration is single chain
  EXPECT_EQ(1, config.num_chains());
  EXPECT_FALSE(config.is_multichain());
  
  // Verify all required pointers are set
  EXPECT_EQ(logger.get(), config.logger());
  EXPECT_EQ(init_writer.get(), config.init_writer());
  EXPECT_EQ(sample_writer.get(), config.sample_writer());
  EXPECT_EQ(diagnostic_writer.get(), config.diagnostic_writer());
  EXPECT_EQ(nullptr, config.metric_writer());
  
  // Verify we can access via chain index methods too
  EXPECT_EQ(init_writer.get(), config.init_writer(0));
  EXPECT_EQ(sample_writer.get(), config.sample_writer(0));
  EXPECT_EQ(diagnostic_writer.get(), config.diagnostic_writer(0));
  EXPECT_EQ(nullptr, config.metric_writer(0));
  
  // Verify we can access via vector getters
  EXPECT_EQ(1, config.init_writers().size());
  EXPECT_EQ(init_writer.get(), config.init_writers()[0]);
  
  // Config should be valid but without metric writer
  EXPECT_TRUE(config.is_valid());
  EXPECT_FALSE(config.has_metric_writer());
}

TEST_F(OutputConfigTest, SingleChainWithMetricConstructor) {
  using config_t = stan::run::output_config<std::stringstream, deleter_noop>;
  using writer_t = stan::callbacks::unique_stream_writer<std::stringstream, deleter_noop>;
  using json_writer_t = stan::callbacks::json_writer<std::stringstream, deleter_noop>;
  
  // Create test objects
  auto init_writer = create_unique_writer();
  auto sample_writer = create_unique_writer();
  auto diagnostic_writer = create_unique_writer();
  auto metric_writer = create_json_writer();
  
  // Create config with all writers including metric writer
  config_t config(logger.get(), init_writer.get(), sample_writer.get(), 
                 diagnostic_writer.get(), metric_writer.get());

  // Verify configuration is single chain
  EXPECT_EQ(1, config.num_chains());
  EXPECT_FALSE(config.is_multichain());
  
  // Verify all writers are set
  EXPECT_EQ(logger.get(), config.logger());
  EXPECT_EQ(init_writer.get(), config.init_writer());
  EXPECT_EQ(sample_writer.get(), config.sample_writer());
  EXPECT_EQ(diagnostic_writer.get(), config.diagnostic_writer());
  EXPECT_EQ(metric_writer.get(), config.metric_writer());
  
  // Config should be valid with metric writer
  EXPECT_TRUE(config.is_valid());
  EXPECT_TRUE(config.has_metric_writer());
}

TEST_F(OutputConfigTest, SingleChainWithNullValues) {
  using config_t = stan::run::output_config<std::stringstream, deleter_noop>;
  
  // Test with only logger provided, other writers null
  config_t config(logger.get());
  
  EXPECT_EQ(logger.get(), config.logger());
  EXPECT_EQ(nullptr, config.init_writer());
  EXPECT_EQ(nullptr, config.sample_writer());
  EXPECT_EQ(nullptr, config.diagnostic_writer());
  EXPECT_EQ(nullptr, config.metric_writer());
  
  // Config should not be valid without required writers
  EXPECT_FALSE(config.is_valid());
}

TEST_F(OutputConfigTest, MultiChainConstructor) {
  using config_t = stan::run::output_config<std::stringstream, deleter_noop>;
  using writer_t = stan::callbacks::unique_stream_writer<std::stringstream, deleter_noop>;
  
  // Create test objects
  const size_t num_chains = 3;
  std::vector<writer_t*> init_writers;
  std::vector<writer_t*> sample_writers;
  std::vector<writer_t*> diagnostic_writers;
  
  std::vector<std::unique_ptr<writer_t>> init_writer_ptrs;
  std::vector<std::unique_ptr<writer_t>> sample_writer_ptrs;
  std::vector<std::unique_ptr<writer_t>> diagnostic_writer_ptrs;
  
  for (size_t i = 0; i < num_chains; ++i) {
    init_writer_ptrs.push_back(create_unique_writer());
    sample_writer_ptrs.push_back(create_unique_writer());
    diagnostic_writer_ptrs.push_back(create_unique_writer());
    
    init_writers.push_back(init_writer_ptrs.back().get());
    sample_writers.push_back(sample_writer_ptrs.back().get());
    diagnostic_writers.push_back(diagnostic_writer_ptrs.back().get());
  }
  
  // Create config with multi-chain constructor
  config_t config(num_chains, logger.get(), init_writers, 
                 sample_writers, diagnostic_writers);

  // Verify configuration is multi-chain
  EXPECT_EQ(num_chains, config.num_chains());
  EXPECT_TRUE(config.is_multichain());
  
  // Verify logger and writer vectors are set
  EXPECT_EQ(logger.get(), config.logger());
  EXPECT_EQ(num_chains, config.init_writers().size());
  EXPECT_EQ(num_chains, config.sample_writers().size());
  EXPECT_EQ(num_chains, config.diagnostic_writers().size());
  EXPECT_TRUE(config.metric_writers().empty());
  
  // Verify individual chain writers are accessible
  for (size_t i = 0; i < num_chains; ++i) {
    EXPECT_EQ(init_writers[i], config.init_writer(i));
    EXPECT_EQ(sample_writers[i], config.sample_writer(i));
    EXPECT_EQ(diagnostic_writers[i], config.diagnostic_writer(i));
    EXPECT_EQ(nullptr, config.metric_writer(i));
  }
  
  // Should not be able to use single-chain getters
  EXPECT_THROW(config.init_writer(), std::logic_error);
  EXPECT_THROW(config.sample_writer(), std::logic_error);
  EXPECT_THROW(config.diagnostic_writer(), std::logic_error);
  EXPECT_THROW(config.metric_writer(), std::logic_error);
  
  // Config should be valid but without metric writers
  EXPECT_TRUE(config.is_valid());
  EXPECT_FALSE(config.has_metric_writer());
}

TEST_F(OutputConfigTest, MultiChainWithMetricConstructor) {
  using config_t = stan::run::output_config<std::stringstream, deleter_noop>;
  using writer_t = stan::callbacks::unique_stream_writer<std::stringstream, deleter_noop>;
  using json_writer_t = stan::callbacks::json_writer<std::stringstream, deleter_noop>;
  
  // Create test objects
  const size_t num_chains = 3;
  std::vector<writer_t*> init_writers;
  std::vector<writer_t*> sample_writers;
  std::vector<writer_t*> diagnostic_writers;
  std::vector<json_writer_t*> metric_writers;
  
  std::vector<std::unique_ptr<writer_t>> init_writer_ptrs;
  std::vector<std::unique_ptr<writer_t>> sample_writer_ptrs;
  std::vector<std::unique_ptr<writer_t>> diagnostic_writer_ptrs;
  std::vector<std::unique_ptr<json_writer_t>> metric_writer_ptrs;
  
  for (size_t i = 0; i < num_chains; ++i) {
    init_writer_ptrs.push_back(create_unique_writer());
    sample_writer_ptrs.push_back(create_unique_writer());
    diagnostic_writer_ptrs.push_back(create_unique_writer());
    metric_writer_ptrs.push_back(create_json_writer());
    
    init_writers.push_back(init_writer_ptrs.back().get());
    sample_writers.push_back(sample_writer_ptrs.back().get());
    diagnostic_writers.push_back(diagnostic_writer_ptrs.back().get());
    metric_writers.push_back(metric_writer_ptrs.back().get());
  }
  
  // Create config with multi-chain constructor including metric writers
  config_t config(num_chains, logger.get(), init_writers, 
                 sample_writers, diagnostic_writers, metric_writers);

  // Verify configuration is multi-chain
  EXPECT_EQ(num_chains, config.num_chains());
  EXPECT_TRUE(config.is_multichain());
  
  // Verify logger and writer vectors are set
  EXPECT_EQ(logger.get(), config.logger());
  EXPECT_EQ(num_chains, config.init_writers().size());
  EXPECT_EQ(num_chains, config.sample_writers().size());
  EXPECT_EQ(num_chains, config.diagnostic_writers().size());
  EXPECT_EQ(num_chains, config.metric_writers().size());
  
  // Verify individual chain writers are accessible
  for (size_t i = 0; i < num_chains; ++i) {
    EXPECT_EQ(init_writers[i], config.init_writer(i));
    EXPECT_EQ(sample_writers[i], config.sample_writer(i));
    EXPECT_EQ(diagnostic_writers[i], config.diagnostic_writer(i));
    EXPECT_EQ(metric_writers[i], config.metric_writer(i));
  }
  
  // Config should be valid with metric writers
  EXPECT_TRUE(config.is_valid());
  EXPECT_TRUE(config.has_metric_writer());
}

TEST_F(OutputConfigTest, FactoryMethods) {
  using config_t = stan::run::output_config<std::stringstream, deleter_noop>;
  using writer_t = stan::callbacks::unique_stream_writer<std::stringstream, deleter_noop>;
  using json_writer_t = stan::callbacks::json_writer<std::stringstream, deleter_noop>;
  
  // Create test objects for single chain
  auto init_writer = create_unique_writer();
  auto sample_writer = create_unique_writer();
  auto diagnostic_writer = create_unique_writer();
  auto metric_writer = create_json_writer();
  
  // Test create factory method - no metric
  auto config1 = config_t::create(
      logger.get(), init_writer.get(), sample_writer.get(), diagnostic_writer.get());
  
  EXPECT_FALSE(config1.is_multichain());
  EXPECT_FALSE(config1.has_metric_writer());
  
  // Test create factory method - with metric
  auto config2 = config_t::create(
      logger.get(), init_writer.get(), sample_writer.get(), 
      diagnostic_writer.get(), metric_writer.get());
  
  EXPECT_FALSE(config2.is_multichain());
  EXPECT_TRUE(config2.has_metric_writer());
  
  // Create test objects for multi-chain
  const size_t num_chains = 2;
  std::vector<writer_t*> init_writers;
  std::vector<writer_t*> sample_writers;
  std::vector<writer_t*> diagnostic_writers;
  std::vector<json_writer_t*> metric_writers;
  
  std::vector<std::unique_ptr<writer_t>> init_writer_ptrs;
  std::vector<std::unique_ptr<writer_t>> sample_writer_ptrs;
  std::vector<std::unique_ptr<writer_t>> diagnostic_writer_ptrs;
  std::vector<std::unique_ptr<json_writer_t>> metric_writer_ptrs;
  
  for (size_t i = 0; i < num_chains; ++i) {
    init_writer_ptrs.push_back(create_unique_writer());
    sample_writer_ptrs.push_back(create_unique_writer());
    diagnostic_writer_ptrs.push_back(create_unique_writer());
    metric_writer_ptrs.push_back(create_json_writer());
    
    init_writers.push_back(init_writer_ptrs.back().get());
    sample_writers.push_back(sample_writer_ptrs.back().get());
    diagnostic_writers.push_back(diagnostic_writer_ptrs.back().get());
    metric_writers.push_back(metric_writer_ptrs.back().get());
  }
  
  // Test create_multi factory method - no metrics
  auto config3 = config_t::create_multi(
      num_chains, logger.get(), init_writers, sample_writers, diagnostic_writers);
  
  EXPECT_TRUE(config3.is_multichain());
  EXPECT_FALSE(config3.has_metric_writer());
  
  // Test create_multi factory method - with metrics
  auto config4 = config_t::create_multi(
      num_chains, logger.get(), init_writers, sample_writers, 
      diagnostic_writers, metric_writers);
  
  EXPECT_TRUE(config4.is_multichain());
  EXPECT_TRUE(config4.has_metric_writer());
}
