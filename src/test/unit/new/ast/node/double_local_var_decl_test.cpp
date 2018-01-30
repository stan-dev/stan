#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

TEST(doubleLocalVarDecl, createVar1) {
  stan::lang::double_local_var_decl x("x");

  // check double_local_var_decl
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_double_type());
  EXPECT_TRUE(is_nil(x.def_));

  // check local_var_decl wrapper
  stan::lang::local_var_decl lvar(x);
  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::double_type());
  EXPECT_FALSE(lvar.has_def());

  std::vector<stan::lang::expression> lvar_sizes = lvar.type().size();
  EXPECT_EQ(lvar_sizes.size(), 0);

}

