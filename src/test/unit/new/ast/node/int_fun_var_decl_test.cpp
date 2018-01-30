#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

TEST(intLocalVarDecl, createVar1) {
  stan::lang::int_fun_var_decl x("x");

  // check int_fun_var_decl
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_int_type());
  EXPECT_TRUE(is_nil(x.def_));
  EXPECT_FALSE(x.is_data_);

  // check fun_var_decl wrapper
  stan::lang::fun_var_decl fvar(x);
  EXPECT_EQ(fvar.name(), "x");
  EXPECT_EQ(fvar.bare_type(), stan::lang::int_type());

  EXPECT_FALSE(fvar.var_decl().is_data());
  fvar.set_is_data();
  EXPECT_TRUE(fvar.var_decl().is_data());
}

