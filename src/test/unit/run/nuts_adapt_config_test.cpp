#include <stan/run/config_nuts_adapt.hpp>
#include <gtest/gtest.h>

TEST(NutsAdaptConfigTest, DefaultConstructor) {
  auto config_nuts_adapt = stan::run::config_nuts_adapt::create().build();
  EXPECT_FLOAT_EQ(stan::run::delta().value(), config_nuts_adapt.delta());
  EXPECT_FLOAT_EQ(stan::run::gamma().value(), config_nuts_adapt.gamma());
  EXPECT_FLOAT_EQ(stan::run::kappa().value(), config_nuts_adapt.kappa());
  EXPECT_FLOAT_EQ(stan::run::t0().value(), config_nuts_adapt.t0());
  EXPECT_EQ(stan::run::init_buffer().value(), config_nuts_adapt.init_buffer());
  EXPECT_EQ(stan::run::term_buffer().value(), config_nuts_adapt.term_buffer());
  EXPECT_EQ(stan::run::window().value(), config_nuts_adapt.window());
}

TEST(McmcIterConfigTest, Constructor_2) {
  auto config_nuts_adapt = stan::run::config_nuts_adapt::create()
    .init_buffer(3).window(5).term_buffer(0).build();
  EXPECT_EQ(3, config_nuts_adapt.init_buffer());
  EXPECT_EQ(0, config_nuts_adapt.term_buffer());
  EXPECT_EQ(5, config_nuts_adapt.window());
}

TEST(McmcIterConfigTest, Constructor_bad) {
  EXPECT_THROW(
	       auto config_nuts_adapt = stan::run::config_nuts_adapt::create()
	       .delta(-3).gamma(-4).kappa(-5).build(),
	       std::invalid_argument);
}

TEST(McmcIterConfigTest, setters_bad) {
  auto config_nuts_adapt = stan::run::config_nuts_adapt::create();
  EXPECT_THROW(config_nuts_adapt.delta(-1), std::invalid_argument);
  EXPECT_THROW(config_nuts_adapt.gamma(-1), std::invalid_argument);
  EXPECT_THROW(config_nuts_adapt.kappa(-1), std::invalid_argument);
}
