#include <stan/run/hmc_nuts_config.hpp>
#include <gtest/gtest.h>

class HmcNutsConfigTest : public ::testing::Test {
protected:
  void SetUp() override {
    json_dir_ = "src/test/unit/run/json/";
    output_dir_ = "test_output";
  }
  
  std::string json_dir_;
  std::string output_dir_;
};

TEST_F(HmcNutsConfigTest, DefaultConstructor) {
  auto config = stan::run::hmc_nuts_config::create()
    .output_dir(output_dir_)
    .build();
  
  // Process defaults
  EXPECT_EQ(1u, config.num_chains());
  EXPECT_EQ(1u, config.seed());
  EXPECT_DOUBLE_EQ(2.0, config.init_radius());
  EXPECT_EQ(output_dir_, config.output_dir());
  EXPECT_FALSE(config.has_init_params());
  
  // HMC defaults
  EXPECT_EQ(1000, config.warmup());
  EXPECT_EQ(1000, config.samples());
  EXPECT_EQ(1, config.thin());
  EXPECT_DOUBLE_EQ(1.0, config.stepsize());
  
  // NUTS adaptation defaults
  EXPECT_DOUBLE_EQ(0.8, config.delta());
  EXPECT_DOUBLE_EQ(0.05, config.gamma());
  EXPECT_EQ(75u, config.init_buffer());
  
  // Output options defaults
  EXPECT_FALSE(config.save_start_params());
  EXPECT_FALSE(config.save_warmup());
  EXPECT_FALSE(config.save_diagnostics());
  EXPECT_FALSE(config.save_metric());
}

TEST_F(HmcNutsConfigTest, CustomValues) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(4)
    .seed(12345)
    .init_radius(1.0)
    .output_dir(output_dir_)
    .warmup(500)
    .samples(2000)
    .thin(2)
    .stepsize(0.1)
    .delta(0.9)
    .gamma(0.1)
    .init_buffer(100)
    .save_warmup(true)
    .save_diagnostics(true)
    .build();
  
  // Process config
  EXPECT_EQ(4u, config.num_chains());
  EXPECT_EQ(12345u, config.seed());
  EXPECT_DOUBLE_EQ(1.0, config.init_radius());
  
  // HMC config
  EXPECT_EQ(500, config.warmup());
  EXPECT_EQ(2000, config.samples());
  EXPECT_EQ(2, config.thin());
  EXPECT_DOUBLE_EQ(0.1, config.stepsize());
  
  // NUTS adaptation
  EXPECT_DOUBLE_EQ(0.9, config.delta());
  EXPECT_DOUBLE_EQ(0.1, config.gamma());
  EXPECT_EQ(100u, config.init_buffer());
  
  // Output options
  EXPECT_FALSE(config.save_start_params());
  EXPECT_TRUE(config.save_warmup());
  EXPECT_TRUE(config.save_diagnostics());
  EXPECT_FALSE(config.save_metric());
}

TEST_F(HmcNutsConfigTest, AccessToComponents) {
  auto config = stan::run::hmc_nuts_config::create()
    .output_dir(output_dir_)
    .build();
  
  // Access individual components
  const auto& process_cfg = config.process();
  const auto& hmc_cfg = config.hmc();
  const auto& adapt_cfg = config.adaptation();
  
  EXPECT_EQ(1u, process_cfg.num_chains());
  EXPECT_EQ(1000, hmc_cfg.warmup());
  EXPECT_DOUBLE_EQ(0.8, adapt_cfg.delta());
}

TEST_F(HmcNutsConfigTest, AllOutputOptions) {
  auto config = stan::run::hmc_nuts_config::create()
    .output_dir(output_dir_)
    .save_start_params(true)
    .save_warmup(true)
    .save_diagnostics(true)
    .save_metric(true)
    .build();
  
  EXPECT_TRUE(config.save_start_params());
  EXPECT_TRUE(config.save_warmup());
  EXPECT_TRUE(config.save_diagnostics());
  EXPECT_TRUE(config.save_metric());
}

TEST_F(HmcNutsConfigTest, InheritedProcessValidation) {
  // Test that process component validation still works
  EXPECT_THROW(
    stan::run::hmc_nuts_config::create()
      .output_dir(output_dir_)
      .num_chains(0)  // Invalid
      .build(),
    std::invalid_argument);
}

TEST_F(HmcNutsConfigTest, InheritedNutsValidation) {
  // Test that NUTS component validation still works
  EXPECT_THROW(
    stan::run::hmc_nuts_config::create()
      .output_dir(output_dir_)
      .delta(1.5)  // Invalid delta > 1
      .build(),
    std::invalid_argument);
}

TEST_F(HmcNutsConfigTest, WithInitParams) {
  std::vector<std::string> init_files = {json_dir_ + "parameters.inits.json"};
  
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .init_params(init_files)
    .output_dir(output_dir_)
    .build();
  
  EXPECT_EQ(1u, config.num_chains());
  EXPECT_TRUE(config.has_init_params());
}

TEST_F(HmcNutsConfigTest, ComprehensiveConfiguration) {
  std::vector<std::string> init_files = {
    json_dir_ + "parameters.inits.json",
    json_dir_ + "parameters.inits.json"
  };
  
  auto config = stan::run::hmc_nuts_config::create()
    // Process configuration
    .num_chains(2)
    .seed(54321)
    .init_radius(0.5)
    .init_params(init_files)
    .output_dir(output_dir_)
    
    // HMC configuration
    .warmup(1500)
    .samples(3000)
    .thin(3)
    .refresh(200)
    .stepsize(0.05)
    .stepsize_jitter(0.2)
    .max_depth(12)
    
    // NUTS adaptation
    .delta(0.85)
    .gamma(0.07)
    .kappa(0.6)
    .t0(15.0)
    .init_buffer(80)
    .term_buffer(60)
    .window(30)
    
    // Output options
    .save_start_params(true)
    .save_warmup(false)
    .save_diagnostics(true)
    .save_metric(true)
    .build();
  
  // Verify all values are set correctly
  EXPECT_EQ(2u, config.num_chains());
  EXPECT_EQ(54321u, config.seed());
  EXPECT_DOUBLE_EQ(0.5, config.init_radius());
  EXPECT_TRUE(config.has_init_params());
  
  EXPECT_EQ(1500, config.warmup());
  EXPECT_EQ(3000, config.samples());
  EXPECT_EQ(3, config.thin());
  
  EXPECT_DOUBLE_EQ(0.85, config.delta());
  EXPECT_DOUBLE_EQ(0.07, config.gamma());
  EXPECT_EQ(80u, config.init_buffer());
  
  EXPECT_TRUE(config.save_start_params());
  EXPECT_FALSE(config.save_warmup());
  EXPECT_TRUE(config.save_diagnostics());
  EXPECT_TRUE(config.save_metric());
}
