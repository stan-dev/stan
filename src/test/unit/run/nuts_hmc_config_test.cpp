#include <stan/callbacks/logger.hpp>
#include <stan/run/nuts_hmc_config.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>

class NutsHmcConfigTest : public ::testing::Test {
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

TEST_F(NutsHmcConfigTest, DefaultConstructor) {
  using stan::run::nuts_hmc_config;
  
  // Create config with default constructor
  nuts_hmc_config config;

  // Verify all defaults are set correctly
  EXPECT_EQ(nuts_hmc_config::metric_t::DIAG_E, config.metric_type());
  EXPECT_EQ(nullptr, config.init_inv_metric());
  EXPECT_FLOAT_EQ(2.0, config.init_radius());
  
  // Test default parameter values
  EXPECT_FLOAT_EQ(1.0, config.stepsize());
  EXPECT_FLOAT_EQ(0.0, config.stepsize_jitter());
  EXPECT_EQ(10, config.max_depth());
  EXPECT_FLOAT_EQ(0.8, config.delta());
  EXPECT_FLOAT_EQ(0.05, config.gamma());
  EXPECT_FLOAT_EQ(0.75, config.kappa());
  EXPECT_FLOAT_EQ(10.0, config.t0());
  EXPECT_EQ(75, config.init_buffer());
  EXPECT_EQ(50, config.term_buffer());
  EXPECT_EQ(25, config.window());
  EXPECT_TRUE(config.adaptation_engaged());
}

TEST_F(NutsHmcConfigTest, FullConstructor) {
  using stan::run::nuts_hmc_config;
  
  // Create config with the full constructor
  nuts_hmc_config config(
      nuts_hmc_config::metric_t::UNIT_E,
      nullptr, //
      1.5,    // init_radius
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
  EXPECT_EQ(nuts_hmc_config::metric_t::UNIT_E, config.metric_type());
  EXPECT_EQ(nullptr, config.init_inv_metric());
  EXPECT_FLOAT_EQ(1.5, config.init_radius());
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

TEST_F(NutsHmcConfigTest, ConfigureSampler) {
  using stan::run::nuts_hmc_config;
  
  // Create default config
  nuts_hmc_config config;
  
  // Configure sampler settings
  config.configure_sampler(
      nuts_hmc_config::metric_t::DENSE_E,
      nullptr,               // init_inv_metric
      3.0,                   // init_radius
      0.25,   // stepsize
      0.1,    // stepsize_jitter
      15      // max_depth
  );
  
  // Verify sampler settings changed
  EXPECT_EQ(nuts_hmc_config::metric_t::DENSE_E, config.metric_type());
  EXPECT_FLOAT_EQ(3.0, config.init_radius());
  EXPECT_FLOAT_EQ(0.25, config.stepsize());
  EXPECT_FLOAT_EQ(0.1, config.stepsize_jitter());
  EXPECT_EQ(15, config.max_depth());
  
  // Adaptation should be disabled
  EXPECT_FALSE(config.adaptation_engaged());
}

TEST_F(NutsHmcConfigTest, ConfigureAdaptation) {
  using stan::run::nuts_hmc_config;
  
  // Create default config
  nuts_hmc_config config;
  
  // Initially adaptation should be enabled (default)
  EXPECT_TRUE(config.adaptation_engaged());
  
  // Disable it by configuring sampler
  config.configure_sampler(nuts_hmc_config::metric_t::UNIT_E);
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

TEST_F(NutsHmcConfigTest, StaticNonAdaptive) {
  using stan::run::nuts_hmc_config;
  
  // Create using static factory method
  auto config = nuts_hmc_config::non_adaptive(
      nuts_hmc_config::metric_t::DIAG_E,
      nullptr,               // init_inv_metric
      1.8,                   // init_radius
      0.75,   // stepsize
      0.05,   // stepsize_jitter
      12      // max_depth
  );
  
  // Verify settings
  EXPECT_EQ(nuts_hmc_config::metric_t::DIAG_E, config.metric_type());
  EXPECT_FLOAT_EQ(1.8, config.init_radius());
  EXPECT_FLOAT_EQ(0.75, config.stepsize());
  EXPECT_FLOAT_EQ(0.05, config.stepsize_jitter());
  EXPECT_EQ(12, config.max_depth());
  
  // Adaptation should be disabled
  EXPECT_FALSE(config.adaptation_engaged());
}

TEST_F(NutsHmcConfigTest, StaticAdaptive) {
  using stan::run::nuts_hmc_config;
  
  // Create using static factory method
  auto config = nuts_hmc_config::adaptive(
      nuts_hmc_config::metric_t::UNIT_E,
      nullptr, // init_inv_metric
      1.5,    // init_radius
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
  EXPECT_EQ(nuts_hmc_config::metric_t::UNIT_E, config.metric_type());
  EXPECT_EQ(nullptr, config.init_inv_metric());
  EXPECT_FLOAT_EQ(1.5, config.init_radius());
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

TEST_F(NutsHmcConfigTest, Setters) {
  using stan::run::nuts_hmc_config;
  
  // Create default config
  nuts_hmc_config config;
  
  // Use setters to modify values
  config.set_metric_type(nuts_hmc_config::metric_t::DENSE_E);
  config.set_init_radius(2.2);
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
  EXPECT_EQ(nuts_hmc_config::metric_t::DENSE_E, config.metric_type());
  EXPECT_FLOAT_EQ(2.2, config.init_radius());
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

TEST_F(NutsHmcConfigTest, ParameterDescriptions) {
  using stan::run::nuts_hmc_config;
  
  // Create default config
  nuts_hmc_config config;
  
  // Check that descriptions are available from the config
  EXPECT_EQ("Initial radius for parameter initialization.", config.init_radius_description());
  EXPECT_EQ("Step size for discrete evolution.", config.stepsize_description());
  EXPECT_EQ("Uniformly random jitter of the stepsize, in percent.", config.stepsize_jitter_description());
  EXPECT_EQ("Maximum tree depth.", config.max_depth_description());
  EXPECT_EQ("Adaptation target acceptance statistic.", config.delta_description());
  EXPECT_EQ("Adaptation regularization scale.", config.gamma_description());
  EXPECT_EQ("Adaptation relaxation exponent.", config.kappa_description());
  EXPECT_EQ("Adaptation iteration offset.", config.t0_description());
  EXPECT_EQ("Width of initial fast adaptation interval.", config.init_buffer_description());
  EXPECT_EQ("Width of final fast adaptation interval.", config.term_buffer_description());
  EXPECT_EQ("Initial width of slow adaptation interval.", config.window_description());
  EXPECT_EQ("Indicates whether adaptation is engaged.", config.adaptation_engaged_description());
}

TEST_F(NutsHmcConfigTest, ParameterValidationOnSetter) {
  using stan::run::nuts_hmc_config;
  
  // Create default config
  nuts_hmc_config config;
  
  // Test setter validation - Invalid values should throw immediately
  EXPECT_THROW(config.set_init_radius(-1.0), std::invalid_argument);
  EXPECT_THROW(config.set_init_radius(0.0), std::invalid_argument);
  EXPECT_THROW(config.set_stepsize(-1.0), std::invalid_argument);
  EXPECT_THROW(config.set_stepsize_jitter(1.5), std::invalid_argument);
  EXPECT_THROW(config.set_max_depth(0), std::invalid_argument);
  EXPECT_THROW(config.set_delta(0.0), std::invalid_argument);
  EXPECT_THROW(config.set_gamma(-0.2), std::invalid_argument);
  EXPECT_THROW(config.set_kappa(-0.5), std::invalid_argument);
  EXPECT_THROW(config.set_t0(-5.0), std::invalid_argument);
}

TEST_F(NutsHmcConfigTest, CheckConsistency) {
  using stan::run::nuts_hmc_config;
  
  // Create config with non-unit metric but no init_inv_metric
  nuts_hmc_config config;
  config.set_metric_type(nuts_hmc_config::metric_t::DIAG_E);
  config.set_init_inv_metric(nullptr);
  
  // Set up logger for consistency check
  TestLogger logger(logger_stream);
  
  // Check consistency (should pass but log a message)
  EXPECT_TRUE(config.check_consistency(logger));
  
  // Logger stream should contain info about null metric
  std::string log_output = logger_stream->str();
  EXPECT_NE(std::string::npos, log_output.find("No initial inverse metric provided"));
}

TEST_F(NutsHmcConfigTest, DefaultValuesMatchParamClasses) {
  // Test that default values in nuts_hmc_config match those in parameter classes
  using stan::run::nuts_hmc_config;
  
  nuts_hmc_config config;
  
  EXPECT_EQ(stan::run::init_radius::default_value(), config.init_radius());
  EXPECT_EQ(stan::run::stepsize::default_value(), config.stepsize());
  EXPECT_EQ(stan::run::stepsize_jitter::default_value(), config.stepsize_jitter());
  EXPECT_EQ(stan::run::max_depth::default_value(), config.max_depth());
  EXPECT_EQ(stan::run::delta::default_value(), config.delta());
  EXPECT_EQ(stan::run::gamma::default_value(), config.gamma());
  EXPECT_EQ(stan::run::kappa::default_value(), config.kappa());
  EXPECT_EQ(stan::run::t0::default_value(), config.t0());
  EXPECT_EQ(stan::run::init_buffer::default_value(), config.init_buffer());
  EXPECT_EQ(stan::run::term_buffer::default_value(), config.term_buffer());
  EXPECT_EQ(stan::run::window::default_value(), config.window());
  EXPECT_EQ(stan::run::adaptation_engaged::default_value(), config.adaptation_engaged());
}
