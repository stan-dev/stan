#include <stan/run/config/nuts_adapt_config.hpp>
#include <gtest/gtest.h>

TEST(NutsAdaptConfigTest, DefaultConstructor) {
  auto nuts_adapt_config = stan::run::nuts_adapt_config::create().build();
  EXPECT_FLOAT_EQ(stan::run::delta().value(), nuts_adapt_config.delta());
  EXPECT_FLOAT_EQ(stan::run::gamma().value(), nuts_adapt_config.gamma());
  EXPECT_FLOAT_EQ(stan::run::kappa().value(), nuts_adapt_config.kappa());
  EXPECT_FLOAT_EQ(stan::run::t0().value(), nuts_adapt_config.t0());
  EXPECT_EQ(stan::run::init_buffer().value(), nuts_adapt_config.init_buffer());
  EXPECT_EQ(stan::run::term_buffer().value(), nuts_adapt_config.term_buffer());
  EXPECT_EQ(stan::run::window().value(), nuts_adapt_config.window());
}

TEST(McmcIterConfigTest, Constructor_2) {
  auto nuts_adapt_config = stan::run::nuts_adapt_config::create()
    .init_buffer(3).window(5).term_buffer(0).build();
  EXPECT_EQ(3, nuts_adapt_config.init_buffer());
  EXPECT_EQ(0, nuts_adapt_config.term_buffer());
  EXPECT_EQ(5, nuts_adapt_config.window());
}

TEST(McmcIterConfigTest, Constructor_bad) {
  EXPECT_THROW(
	       auto nuts_adapt_config = stan::run::nuts_adapt_config::create()
	       .delta(-3).gamma(-4).kappa(-5).build(),
	       std::invalid_argument);
}

TEST(McmcIterConfigTest, setters_bad) {
  auto nuts_adapt_config = stan::run::nuts_adapt_config::create();
  EXPECT_THROW(nuts_adapt_config.delta(-1), std::invalid_argument);
  EXPECT_THROW(nuts_adapt_config.gamma(-1), std::invalid_argument);
  EXPECT_THROW(nuts_adapt_config.kappa(-1), std::invalid_argument);
}
