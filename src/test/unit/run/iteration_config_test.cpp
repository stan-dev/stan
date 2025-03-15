#include <stan/run/iteration_config.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>

class IterationConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up a logger for consistency check tests
    logger_stream = std::make_shared<std::stringstream>();
  }

  class TestLogger : public stan::callbacks::logger {
   public:
    explicit TestLogger(std::shared_ptr<std::stringstream> stream)
        : stream_(stream) {}

    void info(const std::string& message) override { *stream_ << message << std::endl; }
    void error(const std::string& message) override { *stream_ << "ERROR: " << message << std::endl; }
    void warn(const std::string& message) override { *stream_ << "WARNING: " << message << std::endl; }
    void debug(const std::string& message) override { *stream_ << "DEBUG: " << message << std::endl; }
    void fatal(const std::string& message) override { *stream_ << "FATAL: " << message << std::endl; }

    std::shared_ptr<std::stringstream> stream_;
  };

  std::shared_ptr<std::stringstream> logger_stream;
};

TEST_F(IterationConfigTest, DefaultConstructor) {
  using stan::run::iteration_config;
  
  // Create config with default constructor
  iteration_config config;

  // Verify all defaults are set correctly
  EXPECT_EQ(1000, config.num_warmup());
  EXPECT_EQ(1000, config.num_samples());
  EXPECT_FALSE(config.save_warmup());
  EXPECT_EQ(1, config.thin());
  EXPECT_EQ(100, config.refresh());
  
  // Check descriptions
  EXPECT_EQ("Number of warmup iterations.", config.num_warmup_description());
  EXPECT_EQ("Number of sampling iterations.", config.num_samples_description());
  EXPECT_EQ("Save warmup iterations to output.", config.save_warmup_description());
  EXPECT_EQ("Period between saved samples.", config.thin_description());
  EXPECT_EQ("Period between status output messages.", config.refresh_description());
}

TEST_F(IterationConfigTest, FullConstructor) {
  using stan::run::iteration_config;
  
  // Create config with the full constructor
  iteration_config config(
      500,     // num_warmup
      2000,    // num_samples
      true,    // save_warmup
      5,       // thin
      250      // refresh
  );

  // Verify values are set correctly
  EXPECT_EQ(500, config.num_warmup());
  EXPECT_EQ(2000, config.num_samples());
  EXPECT_TRUE(config.save_warmup());
  EXPECT_EQ(5, config.thin());
  EXPECT_EQ(250, config.refresh());
}

TEST_F(IterationConfigTest, CreateFactoryMethod) {
  using stan::run::iteration_config;
  
  // Create using factory method with some custom values
  auto config = iteration_config::create(
      750,     // num_warmup
      1500,    // num_samples
      true,    // save_warmup
      10,      // thin
      200      // refresh
  );
  
  // Verify settings
  EXPECT_EQ(750, config.num_warmup());
  EXPECT_EQ(1500, config.num_samples());
  EXPECT_TRUE(config.save_warmup());
  EXPECT_EQ(10, config.thin());
  EXPECT_EQ(200, config.refresh());
}

TEST_F(IterationConfigTest, CreateFactoryMethodDefaults) {
  using stan::run::iteration_config;
  
  // Create using factory method with defaults
  auto config = iteration_config::create();
  
  // Verify settings match defaults
  EXPECT_EQ(1000, config.num_warmup());
  EXPECT_EQ(1000, config.num_samples());
  EXPECT_FALSE(config.save_warmup());
  EXPECT_EQ(1, config.thin());
  EXPECT_EQ(100, config.refresh());
}

TEST_F(IterationConfigTest, Setters) {
  using stan::run::iteration_config;
  
  // Create default config
  iteration_config config;
  
  // Use setters to modify values
  config.set_num_warmup(250);
  config.set_num_samples(500);
  config.set_save_warmup(true);
  config.set_thin(2);
  config.set_refresh(25);
  
  // Verify values
  EXPECT_EQ(250, config.num_warmup());
  EXPECT_EQ(500, config.num_samples());
  EXPECT_TRUE(config.save_warmup());
  EXPECT_EQ(2, config.thin());
  EXPECT_EQ(25, config.refresh());
}

TEST_F(IterationConfigTest, ParameterValidationOnSetter) {
  using stan::run::iteration_config;
  
  // Create default config
  iteration_config config;
  
  // Test setter validation - Invalid values should throw immediately
  EXPECT_THROW(config.set_num_warmup(-1), std::invalid_argument);
  EXPECT_THROW(config.set_num_samples(-10), std::invalid_argument);
  EXPECT_THROW(config.set_thin(0), std::invalid_argument);
  EXPECT_THROW(config.set_thin(-5), std::invalid_argument);
  
  // No exceptions for refresh since any value is valid
  EXPECT_NO_THROW(config.set_refresh(-1));
}

TEST_F(IterationConfigTest, CheckConsistencyWithHighIterations) {
  using stan::run::iteration_config;
  
  // Create config with very high iteration count
  iteration_config config;
  config.set_num_warmup(500000);
  config.set_num_samples(1000000);
  
  // Set up logger for consistency check
  TestLogger logger(logger_stream);
  
  // Check consistency (should pass but log a warning)
  EXPECT_TRUE(config.check_consistency(logger));
  
  // Logger stream should contain warning about high iteration count
  std::string log_output = logger_stream->str();
  EXPECT_NE(std::string::npos, log_output.find("Total number of iterations is very large"));
}

TEST_F(IterationConfigTest, CheckConsistencyWithHighThinning) {
  using stan::run::iteration_config;
  
  // Create config with high thinning relative to samples
  iteration_config config;
  config.set_num_samples(100);
  config.set_thin(20);
  
  // Set up logger for consistency check
  TestLogger logger(logger_stream);
  
  // Check consistency (should pass but log a warning)
  EXPECT_TRUE(config.check_consistency(logger));
  
  // Logger stream should contain warning about high thinning rate
  std::string log_output = logger_stream->str();
  EXPECT_NE(std::string::npos, log_output.find("Thinning rate"));
  EXPECT_NE(std::string::npos, log_output.find("is large relative to number of samples"));
}

TEST_F(IterationConfigTest, DefaultValuesMatchParamClasses) {
  // Test that default values in iteration_config match those in parameter classes
  using stan::run::iteration_config;
  
  iteration_config config;
  
  EXPECT_EQ(stan::run::num_warmup::default_value(), config.num_warmup());
  EXPECT_EQ(stan::run::num_samples::default_value(), config.num_samples());
  EXPECT_EQ(stan::run::save_warmup::default_value(), config.save_warmup());
  EXPECT_EQ(stan::run::thin::default_value(), config.thin());
  EXPECT_EQ(stan::run::refresh::default_value(), config.refresh());
}
