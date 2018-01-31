#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

TEST(vectorLocalVarDecl, createVar1) {
  stan::lang::expression N(stan::lang::int_literal(1));
  stan::lang::vector_local_type vlt(N);
  stan::lang::vector_local_var_decl x("x", vlt);

  // check vector_local_var_decl
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_vector_type());
  EXPECT_TRUE(is_nil(x.def_));

  // check vector_local_type
  stan::lang::expression x_size = x.type_.N_;
  EXPECT_TRUE(x_size.bare_type().is_int_type());

  // check local_var_decl wrapper
  stan::lang::local_var_decl lvar(x);
  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::vector_type());
  EXPECT_FALSE(lvar.has_def());

  std::vector<stan::lang::expression> lvar_sizes = lvar.type().size();
  EXPECT_EQ(lvar_sizes.size(), 1);
  EXPECT_TRUE(lvar_sizes.at(0).bare_type().is_int_type());
}

