#include <stan/mcmc/base_adapter.hpp>
#include <gtest/gtest.h>

TEST(McmcBaseAdapter, engage_adaptation) {
  stan::mcmc::base_adapter adapter;
  adapter.engage_adaptation();
  EXPECT_TRUE(adapter.adapting());
}

TEST(McmcBaseAdapter, disengage_adaptation) {
  stan::mcmc::base_adapter adapter;
  adapter.disengage_adaptation();
  EXPECT_FALSE(adapter.adapting());
}
