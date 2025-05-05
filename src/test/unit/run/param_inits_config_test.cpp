#include <stan/run/config/param_inits_config.hpp>
#include <gtest/gtest.h>


TEST(ParamInitsConfigTest, DefaultConstructor) {
  auto param_inits_config = stan::run::param_inits_config::create().build();
  EXPECT_EQ(stan::run::init_radius().value(), param_inits_config.init_radius());
  EXPECT_EQ(nullptr, param_inits_config.init_params());
}

TEST(ParamInitsConfigTest, Constructor_2) {
  auto param_inits_config = stan::run::param_inits_config::create()
    .init_radius(5.1).build();
  EXPECT_FLOAT_EQ(5.1, param_inits_config.init_radius());
}

TEST(ParamInitsConfigTest, Constructor_bad) {
  EXPECT_THROW(
	       auto param_inits_config = stan::run::param_inits_config::create()
	       .init_radius(-5.1).build(),
	       std::invalid_argument);
}


TEST(ParamInitsConfigTest, setters_bad) {
  auto param_inits_config = stan::run::param_inits_config::create();
  EXPECT_THROW(param_inits_config.init_radius(-1), std::invalid_argument);
}
