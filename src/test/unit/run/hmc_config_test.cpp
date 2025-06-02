#include <stan/run/hmc_config.hpp>
#include <gtest/gtest.h>

TEST(HmcConfigTest, DefaultConstructor) {
  size_t num_chains = 1;
  auto config = stan::run::hmc_config::create().build(num_chains);
  
  EXPECT_EQ(1000, config.warmup());
  EXPECT_EQ(1000, config.samples());
  EXPECT_EQ(1, config.thin());
  EXPECT_EQ(100, config.refresh());
  EXPECT_DOUBLE_EQ(1.0, config.stepsize());
  EXPECT_DOUBLE_EQ(0.0, config.stepsize_jitter());
  EXPECT_EQ(10, config.max_depth());
  EXPECT_EQ(0, config.init_metric().size());
}

TEST(HmcConfigTest, CustomValues) {
  size_t num_chains = 1;
  auto config = stan::run::hmc_config::create()
    .warmup(500)
    .samples(2000)
    .thin(2)
    .refresh(50)
    .stepsize(0.1)
    .stepsize_jitter(0.1)
    .max_depth(15)
    .build(num_chains);
  
  EXPECT_EQ(500, config.warmup());
  EXPECT_EQ(2000, config.samples());
  EXPECT_EQ(2, config.thin());
  EXPECT_EQ(50, config.refresh());
  EXPECT_DOUBLE_EQ(0.1, config.stepsize());
  EXPECT_DOUBLE_EQ(0.1, config.stepsize_jitter());
  EXPECT_EQ(15, config.max_depth());
}

TEST(HmcConfigTest, InvalidWarmup) {
  size_t num_chains = 1;
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .warmup(-1)
      .build(num_chains),
    std::invalid_argument);
}

TEST(HmcConfigTest, ValidZeroWarmup) {
  size_t num_chains = 1;
  auto config = stan::run::hmc_config::create()
    .warmup(0)
    .build(num_chains);
  
  EXPECT_EQ(0, config.warmup());
}

TEST(HmcConfigTest, InvalidSamples) {
  size_t num_chains = 1;
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .samples(-1)
      .build(num_chains),
    std::invalid_argument);
}

TEST(HmcConfigTest, ValidZeroSamples) {
  size_t num_chains = 1;
  auto config = stan::run::hmc_config::create()
    .samples(0)
    .build(num_chains);
  
  EXPECT_EQ(0, config.samples());
}

TEST(HmcConfigTest, InvalidThinZero) {
  size_t num_chains = 1;
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .thin(0)
      .build(num_chains),
    std::invalid_argument);
}

TEST(HmcConfigTest, InvalidThinNegative) {
  size_t num_chains = 1;
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .thin(-1)
      .build(num_chains),
    std::invalid_argument);
}

TEST(HmcConfigTest, InvalidStepsizeZero) {
  size_t num_chains = 1;
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .stepsize(0.0)
      .build(num_chains),
    std::invalid_argument);
}

TEST(HmcConfigTest, InvalidStepsizeNegative) {
  size_t num_chains = 1;
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .stepsize(-0.1)
      .build(num_chains),
    std::invalid_argument);
}

TEST(HmcConfigTest, InvalidStepsizeJitterTooLow) {
  size_t num_chains = 1;
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .stepsize_jitter(-0.1)
      .build(num_chains),
    std::invalid_argument);
}

TEST(HmcConfigTest, InvalidStepsizeJitterTooHigh) {
  size_t num_chains = 1;
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .stepsize_jitter(1.1)
      .build(num_chains),
    std::invalid_argument);
}

TEST(HmcConfigTest, ValidStepsizeJitterBoundaries) {
  size_t num_chains = 1;
  auto config1 = stan::run::hmc_config::create()
    .stepsize_jitter(0.0)
    .build(num_chains);
  EXPECT_DOUBLE_EQ(0.0, config1.stepsize_jitter());
  
  auto config2 = stan::run::hmc_config::create()
    .stepsize_jitter(1.0)
    .build(num_chains);
  EXPECT_DOUBLE_EQ(1.0, config2.stepsize_jitter());
}

TEST(HmcConfigTest, InvalidMaxDepthZero) {
  size_t num_chains = 1;
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .max_depth(0)
      .build(num_chains),
    std::invalid_argument);
}

TEST(HmcConfigTest, InvalidMaxDepthNegative) {
  size_t num_chains = 1;
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .max_depth(-1)
      .build(num_chains),
    std::invalid_argument);
}

TEST(HmcConfigTest, RefreshNegativeAllowed) {
  size_t num_chains = 1;
  auto config = stan::run::hmc_config::create()
    .refresh(-1)
    .build(num_chains);
  
  EXPECT_EQ(-1, config.refresh());
}
