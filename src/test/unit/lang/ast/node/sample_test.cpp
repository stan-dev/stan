#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>

TEST(StanLangAst, Sample) {
  stan::lang::sample s;
  EXPECT_TRUE(s.is_discrete_ == true || s.is_discrete_ == false);

  stan::lang::expression e = stan::lang::int_literal(3);
  stan::lang::distribution d;
  stan::lang::sample s2(e, d);
  EXPECT_TRUE(s2.is_discrete_ == true || s2.is_discrete_ == false);
}

