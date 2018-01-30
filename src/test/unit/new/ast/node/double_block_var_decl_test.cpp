#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

TEST(doubleBlockVarDecl, createVar1) {
  stan::lang::double_literal real_lb(-2.0);
  stan::lang::double_literal real_ub(2.0);
  stan::lang::expression lb(real_lb);
  stan::lang::expression ub(real_ub);
  stan::lang::range v_bounds(lb, ub);
  stan::lang::double_block_var_decl x("x", v_bounds);

  // check double_bloc_var_decl
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_double_type());
  EXPECT_TRUE(is_nil(x.def_));

  // check double_block_var_type
  EXPECT_TRUE(x.type_.bounds_.has_low());
  EXPECT_TRUE(x.type_.bounds_.has_high());

  // check block_var_decl wrapper
  stan::lang::block_var_decl bvar(x);
  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::double_type());
  EXPECT_FALSE(bvar.has_def());

  EXPECT_TRUE(bvar.type().bounds().has_low());
  EXPECT_TRUE(bvar.type().bounds().has_high());

  std::vector<stan::lang::expression> bvar_sizes = bvar.type().size();
  EXPECT_EQ(bvar_sizes.size(), 0);

}

