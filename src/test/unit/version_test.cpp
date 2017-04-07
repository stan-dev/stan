#include <stan/version.hpp>
#include <gtest/gtest.h>

TEST(Stan, macro) {
  EXPECT_EQ(2, STAN_MAJOR);
  EXPECT_EQ(14, STAN_MINOR);
  EXPECT_EQ(0, STAN_PATCH);
}

TEST(Stan, version) {
  EXPECT_EQ("2", stan::MAJOR_VERSION);
  EXPECT_EQ("14", stan::MINOR_VERSION);
  EXPECT_EQ("0", stan::PATCH_VERSION);
}
