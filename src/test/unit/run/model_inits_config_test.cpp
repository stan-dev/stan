#include <stan/run/config_model_inits.hpp>
#include <gtest/gtest.h>


TEST(ModelInitsConfigTest, DefaultConstructor) {
  auto config_model_inits = stan::run::config_model_inits::create().build();
  EXPECT_EQ(stan::run::init_radius_config().value(), config_model_inits.init_radius());
  EXPECT_EQ(nullptr, config_model_inits.init_params());
}

TEST(ModelInitsConfigTest, Constructor_2) {
  auto config_model_inits = stan::run::config_model_inits::create()
    .init_radius(5.1).build();
  EXPECT_FLOAT_EQ(5.1, config_model_inits.init_radius());
}

TEST(ModelInitsConfigTest, Constructor_bad) {
  EXPECT_THROW(
	       auto config_model_inits = stan::run::config_model_inits::create()
	       .init_radius(-5.1).build(),
	       std::invalid_argument);
}


TEST(ModelInitsConfigTest, setters_bad) {
  auto config_model_inits = stan::run::config_model_inits::create();
  EXPECT_THROW(config_model_inits.init_radius(-1), std::invalid_argument);
}
