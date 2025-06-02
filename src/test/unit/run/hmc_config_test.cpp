#include <stan/run/hmc_config.hpp>
#include <gtest/gtest.h>

TEST(HmcConfigTest, DefaultConstructor) {
  auto config = stan::run::hmc_config::create().build();
  
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
  auto config = stan::run::hmc_config::create()
    .warmup(500)
    .samples(2000)
    .thin(2)
    .refresh(50)
    .stepsize(0.1)
    .stepsize_jitter(0.1)
    .max_depth(15)
    .build();
  
  EXPECT_EQ(500, config.warmup());
  EXPECT_EQ(2000, config.samples());
  EXPECT_EQ(2, config.thin());
  EXPECT_EQ(50, config.refresh());
  EXPECT_DOUBLE_EQ(0.1, config.stepsize());
  EXPECT_DOUBLE_EQ(0.1, config.stepsize_jitter());
  EXPECT_EQ(15, config.max_depth());
}

TEST(HmcConfigTest, InvalidWarmup) {
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .warmup(-1)
      .build(),
    std::invalid_argument);
}

TEST(HmcConfigTest, ValidZeroWarmup) {
  auto config = stan::run::hmc_config::create()
    .warmup(0)
    .build();
  
  EXPECT_EQ(0, config.warmup());
}

TEST(HmcConfigTest, InvalidSamples) {
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .samples(-1)
      .build(),
    std::invalid_argument);
}

TEST(HmcConfigTest, ValidZeroSamples) {
  auto config = stan::run::hmc_config::create()
    .samples(0)
    .build();
  
  EXPECT_EQ(0, config.samples());
}

TEST(HmcConfigTest, InvalidThinZero) {
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .thin(0)
      .build(),
    std::invalid_argument);
}

TEST(HmcConfigTest, InvalidThinNegative) {
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .thin(-1)
      .build(),
    std::invalid_argument);
}

TEST(HmcConfigTest, InvalidStepsizeZero) {
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .stepsize(0.0)
      .build(),
    std::invalid_argument);
}

TEST(HmcConfigTest, InvalidStepsizeNegative) {
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .stepsize(-0.1)
      .build(),
    std::invalid_argument);
}

TEST(HmcConfigTest, InvalidStepsizeJitterTooLow) {
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .stepsize_jitter(-0.1)
      .build(),
    std::invalid_argument);
}

TEST(HmcConfigTest, InvalidStepsizeJitterTooHigh) {
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .stepsize_jitter(1.1)
      .build(),
    std::invalid_argument);
}

TEST(HmcConfigTest, ValidStepsizeJitterBoundaries) {
  auto config1 = stan::run::hmc_config::create()
    .stepsize_jitter(0.0)
    .build();
  EXPECT_DOUBLE_EQ(0.0, config1.stepsize_jitter());
  
  auto config2 = stan::run::hmc_config::create()
    .stepsize_jitter(1.0)
    .build();
  EXPECT_DOUBLE_EQ(1.0, config2.stepsize_jitter());
}

TEST(HmcConfigTest, InvalidMaxDepthZero) {
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .max_depth(0)
      .build(),
    std::invalid_argument);
}

TEST(HmcConfigTest, InvalidMaxDepthNegative) {
  EXPECT_THROW(
    stan::run::hmc_config::create()
      .max_depth(-1)
      .build(),
    std::invalid_argument);
}

TEST(HmcConfigTest, RefreshNegativeAllowed) {
  auto config = stan::run::hmc_config::create()
    .refresh(-1)
    .build();
  
  EXPECT_EQ(-1, config.refresh());
}

TEST(HmcConfigTest, ValidateMetricFilenames) {
  std::string json_dir = "src/test/unit/run/json/";
  std::vector<std::string> filenames;
  filenames.push_back(json_dir + "valid_data.json");
  auto config_builder = stan::run::hmc_config::create()
    .init_metric(filenames);
  config_builder.validate(4);
  auto config = config_builder.build();
  EXPECT_EQ(4, config.init_metric().size());
}
