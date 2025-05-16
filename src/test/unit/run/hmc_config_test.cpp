#include <stan/run/config_hmc.hpp>
#include <stan/run/metric_type.hpp>
#include <gtest/gtest.h>


TEST(HmcConfigTest, DefaultConstructor) {
  auto config_hmc = stan::run::config_hmc::create().build();
  EXPECT_EQ(stan::run::num_warmup().value(), config_hmc.num_warmup());
  EXPECT_EQ(stan::run::num_samples().value(), config_hmc.num_samples());
  EXPECT_EQ(stan::run::save_warmup().value(), config_hmc.save_warmup());
  EXPECT_EQ(stan::run::thin().value(), config_hmc.thin());
  EXPECT_EQ(stan::run::refresh().value(), config_hmc.refresh());
  EXPECT_FLOAT_EQ(stan::run::stepsize().value(), config_hmc.stepsize());
  EXPECT_FLOAT_EQ(stan::run::stepsize_jitter().value(), config_hmc.stepsize_jitter());
  EXPECT_EQ(stan::run::max_depth().value(), config_hmc.max_depth());
  EXPECT_EQ(stan::run::metric_t::DIAG_E, config_hmc.metric_type());
  EXPECT_EQ(nullptr, config_hmc.init_inv_metric());
}

TEST(HmcConfigTest, Constructor_2) {
  auto config_hmc = stan::run::config_hmc::create()
    .metric_type(stan::run::metric_t::DENSE_E)
    .stepsize(0.1)
    .max_depth(11)
    .build();
  EXPECT_FLOAT_EQ(0.1, config_hmc.stepsize());
  EXPECT_FLOAT_EQ(stan::run::stepsize_jitter().value(), config_hmc.stepsize_jitter());
  EXPECT_EQ(11, config_hmc.max_depth());
  EXPECT_EQ(stan::run::metric_t::DENSE_E, config_hmc.metric_type());
}

TEST(HmcConfigTest, Constructor_bad) {
  EXPECT_THROW(auto config_hmc = stan::run::config_hmc::create()
	       .max_depth(-11).build(),
	       std::invalid_argument);
  EXPECT_THROW(auto config_hmc = stan::run::config_hmc::create()
	       .stepsize(-0.1).build(),
	       std::invalid_argument);
  EXPECT_THROW(
	       auto config_hmc = stan::run::config_hmc::create()
	       .num_warmup(-5).build(),
	       std::invalid_argument);
  EXPECT_THROW(
	       auto config_hmc = stan::run::config_hmc::create()
	       .num_samples(-10).build(),
	       std::invalid_argument);
  EXPECT_THROW(
	       auto config_hmc = stan::run::config_hmc::create()
	       .thin(-2).build(),
	       std::invalid_argument);
}

TEST(HmcConfigTest, setters_bad) {
  auto config_hmc = stan::run::config_hmc::create();
  EXPECT_THROW(config_hmc.num_warmup(-1), std::invalid_argument);
  EXPECT_THROW(config_hmc.num_samples(-1), std::invalid_argument);
  EXPECT_THROW(config_hmc.thin(-1), std::invalid_argument);
  EXPECT_THROW(config_hmc.stepsize(0.-1), std::invalid_argument);
  EXPECT_THROW(config_hmc.stepsize_jitter(0.-1), std::invalid_argument);
  EXPECT_THROW(config_hmc.max_depth(-1), std::invalid_argument);
}
