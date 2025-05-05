#include <stan/run/config/hmc_config.hpp>
#include <stan/run/metric_type.hpp>
#include <gtest/gtest.h>


TEST(HmcConfigTest, DefaultConstructor) {
  auto hmc_config = stan::run::hmc_config::create().build();
  EXPECT_FLOAT_EQ(stan::run::stepsize().value(), hmc_config.stepsize());
  EXPECT_FLOAT_EQ(stan::run::stepsize_jitter().value(), hmc_config.stepsize_jitter());
  EXPECT_EQ(stan::run::max_depth().value(), hmc_config.max_depth());
  EXPECT_EQ(stan::run::metric_t::DIAG_E, hmc_config.metric_type());
  EXPECT_EQ(nullptr, hmc_config.init_inv_metric());
}

TEST(HmcConfigTest, Constructor_2) {
  auto hmc_config = stan::run::hmc_config::create()
    .metric_type(stan::run::metric_t::DENSE_E)
    .stepsize(0.1)
    .max_depth(11)
    .build();
  EXPECT_FLOAT_EQ(0.1, hmc_config.stepsize());
  EXPECT_FLOAT_EQ(stan::run::stepsize_jitter().value(), hmc_config.stepsize_jitter());
  EXPECT_EQ(11, hmc_config.max_depth());
  EXPECT_EQ(stan::run::metric_t::DENSE_E, hmc_config.metric_type());
}

TEST(HmcConfigTest, Constructor_bad) {
  auto hmc_config = stan::run::hmc_config::create();
  EXPECT_THROW(hmc_config.stepsize(0.-1), std::invalid_argument);
  EXPECT_THROW(hmc_config.stepsize_jitter(0.-1), std::invalid_argument);
  EXPECT_THROW(hmc_config.max_depth(-1), std::invalid_argument);
}
