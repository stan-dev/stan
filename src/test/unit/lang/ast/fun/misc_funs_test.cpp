#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>

TEST(StanLangAstFun, is_space) {
  using stan::lang::is_space;
  EXPECT_TRUE(is_space(' '));
  EXPECT_TRUE(is_space('\n'));
  EXPECT_TRUE(is_space('\t'));
  EXPECT_TRUE(is_space('\r'));

  EXPECT_FALSE(is_space('a'));
  EXPECT_FALSE(is_space('2'));
}

TEST(StanLangAstFun, is_nonempty) {
  using stan::lang::is_nonempty;
  EXPECT_FALSE(is_nonempty(" "));
  EXPECT_FALSE(is_nonempty("\n"));
  EXPECT_FALSE(is_nonempty("\t"));
  EXPECT_FALSE(is_nonempty("\r"));
  EXPECT_FALSE(is_nonempty("   \r  \t  "));

  EXPECT_TRUE(is_nonempty("1"));
  EXPECT_TRUE(is_nonempty("  \r\n \n 1  \n"));
}
