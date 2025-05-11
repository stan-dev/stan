#include <stan/run/config_mcmc_iter.hpp>
#include <gtest/gtest.h>


TEST(McmcIterConfigTest, DefaultConstructor) {
  auto config_mcmc_iter = stan::run::config_mcmc_iter::create().build();
  EXPECT_EQ(stan::run::num_warmup().value(), config_mcmc_iter.num_warmup());
  EXPECT_EQ(stan::run::num_samples().value(), config_mcmc_iter.num_samples());
  EXPECT_EQ(stan::run::save_warmup().value(), config_mcmc_iter.save_warmup());
  EXPECT_EQ(stan::run::thin().value(), config_mcmc_iter.thin());
  EXPECT_EQ(stan::run::refresh().value(), config_mcmc_iter.refresh());
}

TEST(McmcIterConfigTest, Constructor_2) {
  auto config_mcmc_iter = stan::run::config_mcmc_iter::create()
    .num_warmup(5).num_samples(10).save_warmup(false).refresh(2)
    .build();
  EXPECT_EQ(5, config_mcmc_iter.num_warmup());
  EXPECT_EQ(10, config_mcmc_iter.num_samples());
  EXPECT_EQ(false, config_mcmc_iter.save_warmup());
  EXPECT_EQ(stan::run::thin().value(), config_mcmc_iter.thin());
  EXPECT_EQ(2, config_mcmc_iter.refresh());
}

TEST(McmcIterConfigTest, Constructor_bad) {
  EXPECT_THROW(
	       auto config_mcmc_iter = stan::run::config_mcmc_iter::create()
	       .num_warmup(-5).num_samples(-10).thin(-2).build(),
	       std::invalid_argument);
}


TEST(McmcIterConfigTest, setters_bad) {
  auto config_mcmc_iter = stan::run::config_mcmc_iter::create();
  EXPECT_THROW(config_mcmc_iter.num_warmup(-1), std::invalid_argument);
  EXPECT_THROW(config_mcmc_iter.num_samples(-1), std::invalid_argument);
  EXPECT_THROW(config_mcmc_iter.thin(-1), std::invalid_argument);
}

