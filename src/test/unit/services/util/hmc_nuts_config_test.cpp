#include <stan/services/util/hmc_nuts_config.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>

class HmcNutsConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up a logger for validation tests
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

TEST_F(HmcNutsConfigTest, DefaultConstructor) {
  using stan::services::util::hmc_nuts_config;
  using namespace stan::services::util;

  // Create config with default constructor
  hmc_nuts_config config;

  // Verify all defaults are set correctly
  EXPECT_EQ(hmc_nuts_config::metric_t::DIAG_E, config.metric_type());
  EXPECT_EQ(nullptr, config.init_inv_metric());
  EXPECT_FLOAT_EQ(stepsize::default_value(), config.stepsize());
  EXPECT_FLOAT_EQ(stepsize_jitter::default_value(), config.stepsize_jitter());
  EXPECT_EQ(max_depth::default_value(), config.max_depth());
  EXPECT_FLOAT_EQ(delta::default_value(), config.delta());
  EXPECT_FLOAT_EQ(gamma::default_value(), config.gamma());
  EXPECT_FLOAT_EQ(kappa::default_value(), config.kappa());
  EXPECT_FLOAT_EQ(t0::default_value(), config.t0());
  EXPECT_EQ(init_buffer::default_value(), config.init_buffer());
  EXPECT_EQ(term_buffer::default_value(), config.term_buffer());
  EXPECT_EQ(window::default_value(), config.window());
  EXPECT_EQ(adaptation_engaged::default_value(), config.adaptation_engaged());
}

TEST_F(HmcNutsConfigTest, FullConstructor) {
  using stan::services::util::hmc_nuts_config;
  
  // Create config with the full constructor
  hmc_nuts_config config(
      hmc_nuts_config::metric_t::UNIT_E,
      0.5,    // stepsize
      0.2,    // stepsize_jitter
      8,      // max_depth
      0.65,   // delta
      0.1,    // gamma
      0.8,    // kappa
      12.0,   // t0
      80,     // init_buffer
      60,     // term_buffer
      30,     // window
      true    // adaptation_engaged
  );

  // Verify values are set correctly
  EXPECT_EQ(hmc_nuts_config::metric_t::UNIT_E, config.metric_type());
  EXPECT_EQ(nullptr, config.init_inv_metric());
  EXPECT_FLOAT_EQ(0.5, config.stepsize());
  EXPECT_FLOAT_EQ(0.2, config.stepsize_jitter());
  EXPECT_EQ(8, config.max_depth());
  EXPECT_FLOAT_EQ(0.65, config.delta());
  EXPECT_FLOAT_EQ(0.1, config.gamma());
  EXPECT_FLOAT_EQ(0.8, config.kappa());
  EXPECT_FLOAT_EQ(12.0, config.t0());
  EXPECT_EQ(80, config.init_buffer());
  EXPECT_EQ(60, config.term_buffer());
  EXPECT_EQ(30, config.window());
  EXPECT_TRUE(config.adaptation_engaged());
}

TEST_F(HmcNutsConfigTest, ConfigureSampler) {
  using stan::services::util::hmc_nuts_config;
  
  // Create default config
  hmc_nuts_config config;
  
  // Configure sampler settings
  config.configure_sampler(
      hmc_nuts_config::metric_t::DENSE_E,
      0.25,   // stepsize
      0.1,    // stepsize_jitter
      15      // max_depth
  );
  
  // Verify sampler settings changed
  EXPECT_EQ(hmc_nuts_config::metric_t::DENSE_E, config.metric_type());
  EXPECT_FLOAT_EQ(0.25, config.stepsize());
  EXPECT_FLOAT_EQ(0.1, config.stepsize_jitter());
  EXPECT_EQ(15, config.max_depth());
  
  // Adaptation should be disabled
  EXPECT_FALSE(config.adaptation_engaged());
}

TEST_F(HmcNutsConfigTest, ConfigureAdaptation) {
  using stan::services::util::hmc_nuts_config;
  
  // Create default config
  hmc_nuts_config config;
  
  // Initially adaptation should be enabled (default)
  EXPECT_TRUE(config.adaptation_engaged());
  
  // Disable it by configuring sampler
  config.configure_sampler(hmc_nuts_config::metric_t::UNIT_E);
  EXPECT_FALSE(config.adaptation_engaged());
  
  // Re-enable and configure adaptation
  config.configure_adaptation(
      0.9,    // delta
      0.2,    // gamma
      0.6,    // kappa
      5.0,    // t0
      100,    // init_buffer
      75,     // term_buffer
      40      // window
  );
  
  // Verify adaptation settings
  EXPECT_TRUE(config.adaptation_engaged());
  EXPECT_FLOAT_EQ(0.9, config.delta());
  EXPECT_FLOAT_EQ(0.2, config.gamma());
  EXPECT_FLOAT_EQ(0.6, config.kappa());
  EXPECT_FLOAT_EQ(5.0, config.t0());
  EXPECT_EQ(100, config.init_buffer());
  EXPECT_EQ(75, config.term_buffer());
  EXPECT_EQ(40, config.window());
}

TEST_F(HmcNutsConfigTest, StaticNonAdaptive) {
  using stan::services::util::hmc_nuts_config;
  
  // Create using static factory method
  auto config = hmc_nuts_config::non_adaptive(
      hmc_nuts_config::metric_t::DIAG_E,
      0.75,   // stepsize
      0.05,   // stepsize_jitter
      12      // max_depth
  );
  
  // Verify settings
  EXPECT_EQ(hmc_nuts_config::metric_t::DIAG_E, config.metric_type());
  EXPECT_FLOAT_EQ(0.75, config.stepsize());
  EXPECT_FLOAT_EQ(0.05, config.stepsize_jitter());
  EXPECT_EQ(12, config.max_depth());
  
  // Adaptation should be disabled
  EXPECT_FALSE(config.adaptation_engaged());
}

TEST_F(HmcNutsConfigTest, StaticAdaptive) {
  using stan::services::util::hmc_nuts_config;
  
  // Create using static factory method
  auto config = hmc_nuts_config::adaptive(
      hmc_nuts_config::metric_t::UNIT_E,
      0.5,    // stepsize
      0.1,    // stepsize_jitter
      8,      // max_depth
      0.85,   // delta
      0.075,  // gamma
      0.8,    // kappa
      15.0,   // t0
      50,     // init_buffer
      40,     // term_buffer
      20      // window
  );
  
  // Verify settings
  EXPECT_EQ(hmc_nuts_config::metric_t::UNIT_E, config.metric_type());
  EXPECT_FLOAT_EQ(0.5, config.stepsize());
  EXPECT_FLOAT_EQ(0.1, config.stepsize_jitter());
  EXPECT_EQ(8, config.max_depth());
  EXPECT_FLOAT_EQ(0.85, config.delta());
  EXPECT_FLOAT_EQ(0.075, config.gamma());
  EXPECT_FLOAT_EQ(0.8, config.kappa());
  EXPECT_FLOAT_EQ(15.0, config.t0());
  EXPECT_EQ(50, config.init_buffer());
  EXPECT_EQ(40, config.term_buffer());
  EXPECT_EQ(20, config.window());
  
  // Adaptation should be enabled
  EXPECT_TRUE(config.adaptation_engaged());
}

TEST_F(HmcNutsConfigTest, Setters) {
  using stan::services::util::hmc_nuts_config;
  
  // Create default config
  hmc_nuts_config config;
  
  // Use setters to modify values
  config.set_metric_type(hmc_nuts_config::metric_t::DENSE_E);
  config.set_stepsize(0.33);
  config.set_stepsize_jitter(0.15);
  config.set_max_depth(7);
  config.set_delta(0.95);
  config.set_gamma(0.3);
  config.set_kappa(0.5);
  config.set_t0(20.0);
  config.set_init_buffer(60);
  config.set_term_buffer(45);
  config.set_window(35);
  config.set_adaptation_engaged(false);
  
  // Verify values
  EXPECT_EQ(hmc_nuts_config::metric_t::DENSE_E, config.metric_type());
  EXPECT_FLOAT_EQ(0.33, config.stepsize());
  EXPECT_FLOAT_EQ(0.15, config.stepsize_jitter());
  EXPECT_EQ(7, config.max_depth());
  EXPECT_FLOAT_EQ(0.95, config.delta());
  EXPECT_FLOAT_EQ(0.3, config.gamma());
  EXPECT_FLOAT_EQ(0.5, config.kappa());
  EXPECT_FLOAT_EQ(20.0, config.t0());
  EXPECT_EQ(60, config.init_buffer());
  EXPECT_EQ(45, config.term_buffer());
  EXPECT_EQ(35, config.window());
  EXPECT_FALSE(config.adaptation_engaged());
}

TEST_F(HmcNutsConfigTest, ValidateValid) {
  using stan::services::util::hmc_nuts_config;
  
  // Create config with valid values
  hmc_nuts_config config;
  
  // Set up logger for validation
  TestLogger logger(logger_stream);
  
  // Validate (should pass)
  EXPECT_TRUE(config.validate(logger));
  
  // Logger stream should not contain errors
  std::string log_output = logger_stream->str();
  EXPECT_EQ(std::string::npos, log_output.find("ERROR"));
}

TEST_F(HmcNutsConfigTest, ValidateInvalid) {
  using stan::services::util::hmc_nuts_config;
  
  // Create config with invalid values
  hmc_nuts_config config;
  config.set_stepsize(-1.0);  // Invalid (must be > 0)
  
  // Set up logger for validation
  TestLogger logger(logger_stream);
  
  // Validate (should fail)
  EXPECT_FALSE(config.validate(logger));
  
  // Logger stream should contain error about stepsize
  std::string log_output = logger_stream->str();
  EXPECT_NE(std::string::npos, log_output.find("ERROR"));
  EXPECT_NE(std::string::npos, log_output.find("stepsize"));
}

TEST_F(HmcNutsConfigTest, ValidateMultipleInvalid) {
  using stan::services::util::hmc_nuts_config;
  
  // Create config with multiple invalid values
  hmc_nuts_config config;
  config.set_stepsize(-1.0);          // Invalid (must be > 0)
  config.set_stepsize_jitter(1.5);    // Invalid (must be between 0 and 1)
  config.set_max_depth(0);            // Invalid (must be > 0)
  
  // Configure adaptation with invalid values
  config.configure_adaptation(
      1.5,    // delta - Invalid (must be between 0 and 1)
      -0.2,   // gamma - Invalid (must be > 0)
      -0.5,   // kappa - Invalid (must be > 0)
      -5.0    // t0 - Invalid (must be > 0)
  );
  
  // Set up logger for validation
  TestLogger logger(logger_stream);
  
  // Validate (should fail)
  EXPECT_FALSE(config.validate(logger));
  
  // Logger stream should contain multiple errors
  std::string log_output = logger_stream->str();
  EXPECT_NE(std::string::npos, log_output.find("stepsize"));
  EXPECT_NE(std::string::npos, log_output.find("stepsize_jitter"));
  EXPECT_NE(std::string::npos, log_output.find("max_depth"));
  EXPECT_NE(std::string::npos, log_output.find("delta"));
  EXPECT_NE(std::string::npos, log_output.find("gamma"));
  EXPECT_NE(std::string::npos, log_output.find("kappa"));
  EXPECT_NE(std::string::npos, log_output.find("t0"));
}

TEST_F(HmcNutsConfigTest, NullInitMetricLog) {
  using stan::services::util::hmc_nuts_config;
  
  // Create config with non-unit metric but no init_inv_metric
  hmc_nuts_config config;
  config.set_metric_type(hmc_nuts_config::metric_t::DIAG_E);
  config.set_init_inv_metric(nullptr);
  
  // Set up logger for validation
  TestLogger logger(logger_stream);
  
  // Validate (should still pass but log a message)
  EXPECT_TRUE(config.validate(logger));
  
  // Logger stream should contain info about null metric
  std::string log_output = logger_stream->str();
  EXPECT_NE(std::string::npos, log_output.find("No initial inverse metric provided"));
}
