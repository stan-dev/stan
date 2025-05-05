#include <stan/run/config/mcmc_iter_config.hpp>
#include <gtest/gtest.h>


TEST(McmcIterConfigTest, DefaultConstructor) {
  auto mcmc_iter_config = stan::run::mcmc_iter_config::create().build();
  EXPECT_EQ(stan::run::num_warmup().value(), mcmc_iter_config.num_warmup());
  EXPECT_EQ(stan::run::num_samples().value(), mcmc_iter_config.num_samples());
  EXPECT_EQ(stan::run::save_warmup().value(), mcmc_iter_config.save_warmup());
  EXPECT_EQ(stan::run::thin().value(), mcmc_iter_config.thin());
  EXPECT_EQ(stan::run::refresh().value(), mcmc_iter_config.refresh());
}

TEST(McmcIterConfigTest, Constructor_2) {
  auto mcmc_iter_config = stan::run::mcmc_iter_config::create()
    .num_warmup(5).num_samples(10).save_warmup(false).refresh(2)
    .build();
  EXPECT_EQ(5, mcmc_iter_config.num_warmup());
  EXPECT_EQ(10, mcmc_iter_config.num_samples());
  EXPECT_EQ(false, mcmc_iter_config.save_warmup());
  EXPECT_EQ(stan::run::thin().value(), mcmc_iter_config.thin());
  EXPECT_EQ(2, mcmc_iter_config.refresh());
}

TEST(McmcIterConfigTest, Constructor_bad) {
  EXPECT_THROW(
	       auto mcmc_iter_config = stan::run::mcmc_iter_config::create()
	       .num_warmup(-5).num_samples(-10).thin(-2).build(),
	       std::invalid_argument);
}


TEST(McmcIterConfigTest, setters_bad) {
  auto mcmc_iter_config = stan::run::mcmc_iter_config::create();
  EXPECT_THROW(mcmc_iter_config.num_warmup(-1), std::invalid_argument);
  EXPECT_THROW(mcmc_iter_config.num_samples(-1), std::invalid_argument);
  EXPECT_THROW(mcmc_iter_config.thin(-1), std::invalid_argument);
}

