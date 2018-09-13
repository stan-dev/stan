#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>

TEST(langAst, hasRngSuffix) {
  EXPECT_TRUE(stan::lang::has_rng_suffix("foo_rng"));
  EXPECT_FALSE(stan::lang::has_rng_suffix("foo.rng"));
  EXPECT_FALSE(stan::lang::has_rng_suffix("foo.bar"));
}

TEST(langAst, hasLpSuffix) {
  EXPECT_TRUE(stan::lang::has_lp_suffix("foo_lp"));
  EXPECT_FALSE(stan::lang::has_lp_suffix("foo.lp"));
  EXPECT_FALSE(stan::lang::has_lp_suffix("foo.bar"));
}
