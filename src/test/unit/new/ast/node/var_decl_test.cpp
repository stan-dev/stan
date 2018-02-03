#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

TEST(varDecl, create1Arg) {
  stan::lang::var_decl x("x");
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_ill_formed_type());
  EXPECT_TRUE(is_nil(x.def_));
  EXPECT_FALSE(x.bare_type_.is_data());
}

TEST(varDecl, create2Arg) {
  stan::lang::var_decl x("x", stan::lang::int_type());
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_int_type());
  EXPECT_FALSE(x.bare_type_.is_data());
  EXPECT_TRUE(is_nil(x.def_));
}

TEST(varDecl, create3Arg) {
  stan::lang::expression e(stan::lang::int_literal(1));
  stan::lang::var_decl x("x", stan::lang::int_type(), e);
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_int_type());
  EXPECT_FALSE(x.bare_type_.is_data());
  EXPECT_EQ(x.def_.bare_type(), x.bare_type_);
}

TEST(varDecl, createArray) {
  stan::lang::matrix_type tMat;
  stan::lang::bare_array_type d1(tMat);
  stan::lang::var_decl x("x", d1);
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_array_type());
  EXPECT_FALSE(x.bare_type_.is_data());
}

TEST(varDecl, createEmpty) {
  stan::lang::var_decl x;
  EXPECT_EQ(x.name_, "");
  EXPECT_TRUE(x.bare_type_.is_ill_formed_type());
  EXPECT_TRUE(is_nil(x.def_));
  EXPECT_FALSE(x.bare_type_.is_data());
}
