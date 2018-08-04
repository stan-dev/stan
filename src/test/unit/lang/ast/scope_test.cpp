#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>

TEST(StanLangAst, Scope) {
  stan::lang::scope s;
  EXPECT_TRUE(s.is_local() == true || s.is_local() == false);

  stan::lang::scope s2(stan::lang::data_origin);
  EXPECT_TRUE(s2.is_local() == true || s2.is_local() == false);
}
