#include <stan/run/nuts_adapt_config.hpp>
#include <gtest/gtest.h>

TEST(NutsAdaptConfigTest, DefaultConstructor) {
  auto config = stan::run::nuts_adapt_config::create().build();
  
  EXPECT_DOUBLE_EQ(0.8, config.delta());
  EXPECT_DOUBLE_EQ(0.05, config.gamma());
  EXPECT_DOUBLE_EQ(0.75, config.kappa());
  EXPECT_DOUBLE_EQ(10.0, config.t0());
  EXPECT_EQ(75u, config.init_buffer());
  EXPECT_EQ(50u, config.term_buffer());
  EXPECT_EQ(25u, config.window());
}

TEST(NutsAdaptConfigTest, CustomValues) {
  auto config = stan::run::nuts_adapt_config::create()
    .delta(0.9)
    .gamma(0.1)
    .kappa(0.5)
    .t0(20.0)
    .init_buffer(100)
    .term_buffer(75)
    .window(50)
    .build();
  
  EXPECT_DOUBLE_EQ(0.9, config.delta());
  EXPECT_DOUBLE_EQ(0.1, config.gamma());
  EXPECT_DOUBLE_EQ(0.5, config.kappa());
  EXPECT_DOUBLE_EQ(20.0, config.t0());
  EXPECT_EQ(100u, config.init_buffer());
  EXPECT_EQ(75u, config.term_buffer());
  EXPECT_EQ(50u, config.window());
}

TEST(NutsAdaptConfigTest, InvalidDeltaTooLow) {
  EXPECT_THROW(
    stan::run::nuts_adapt_config::create()
      .delta(0.0)
      .build(),
    std::invalid_argument);
}

TEST(NutsAdaptConfigTest, InvalidDeltaTooHigh) {
  EXPECT_THROW(
    stan::run::nuts_adapt_config::create()
      .delta(1.0)
      .build(),
    std::invalid_argument);
}

TEST(NutsAdaptConfigTest, InvalidDeltaNegative) {
  EXPECT_THROW(
    stan::run::nuts_adapt_config::create()
      .delta(-0.1)
      .build(),
    std::invalid_argument);
}

TEST(NutsAdaptConfigTest, InvalidGamma) {
  EXPECT_THROW(
    stan::run::nuts_adapt_config::create()
      .gamma(0.0)
      .build(),
    std::invalid_argument);
  
  EXPECT_THROW(
    stan::run::nuts_adapt_config::create()
      .gamma(-0.1)
      .build(),
    std::invalid_argument);
}

TEST(NutsAdaptConfigTest, InvalidKappa) {
  EXPECT_THROW(
    stan::run::nuts_adapt_config::create()
      .kappa(0.0)
      .build(),
    std::invalid_argument);
  
  EXPECT_THROW(
    stan::run::nuts_adapt_config::create()
      .kappa(-0.1)
      .build(),
    std::invalid_argument);
}

TEST(NutsAdaptConfigTest, InvalidT0) {
  EXPECT_THROW(
    stan::run::nuts_adapt_config::create()
      .t0(0.0)
      .build(),
    std::invalid_argument);
  
  EXPECT_THROW(
    stan::run::nuts_adapt_config::create()
      .t0(-1.0)
      .build(),
    std::invalid_argument);
}

TEST(NutsAdaptConfigTest, ValidBoundaryValues) {
  // Test valid boundary values
  auto config = stan::run::nuts_adapt_config::create()
    .delta(0.01)  // Very small but valid
    .gamma(0.001) // Very small but valid
    .kappa(0.001) // Very small but valid
    .t0(0.001)    // Very small but valid
    .build();
  
  EXPECT_DOUBLE_EQ(0.01, config.delta());
  EXPECT_DOUBLE_EQ(0.001, config.gamma());
  EXPECT_DOUBLE_EQ(0.001, config.kappa());
  EXPECT_DOUBLE_EQ(0.001, config.t0());
}

TEST(NutsAdaptConfigTest, ValidUpperBoundaryDelta) {
  auto config = stan::run::nuts_adapt_config::create()
    .delta(0.99)  // Close to 1 but valid
    .build();
  
  EXPECT_DOUBLE_EQ(0.99, config.delta());
}
