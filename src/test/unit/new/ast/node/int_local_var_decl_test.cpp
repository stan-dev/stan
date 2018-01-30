#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

TEST(intLocalVarDecl, createVar1) {
  stan::lang::int_local_var_decl x("x");

  // check int_local_var_decl
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_int_type());
  EXPECT_TRUE(is_nil(x.def_));

  // check local_var_decl wrapper
  stan::lang::local_var_decl lvar(x);
  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::int_type());
  EXPECT_FALSE(lvar.has_def());

  std::vector<stan::lang::expression> lvar_sizes = lvar.type().size();
  EXPECT_EQ(lvar_sizes.size(), 0);

}

