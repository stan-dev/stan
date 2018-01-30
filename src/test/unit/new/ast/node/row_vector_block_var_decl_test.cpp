#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

TEST(rowVectorBlockVarDecl, createVar1) {
  stan::lang::double_literal real_lb(-2.0);
  stan::lang::double_literal real_ub(2.0);
  stan::lang::expression lb(real_lb);
  stan::lang::expression ub(real_ub);
  stan::lang::range vec_bounds(lb, ub);
  stan::lang::int_literal int_len(5);
  stan::lang::expression vec_len(int_len);
  stan::lang::row_vector_block_var_decl x("x", vec_bounds, vec_len);

  // check row_vector_bloc_var_decl
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_row_vector_type());
  EXPECT_TRUE(is_nil(x.def_));

  // check row_vector_block_var_type
  EXPECT_TRUE(x.type_.bounds_.has_low());
  EXPECT_TRUE(x.type_.bounds_.has_high());
  stan::lang::expression x_size = x.type_.N_;
  EXPECT_TRUE(x_size.bare_type().is_int_type());

  // check block_var_decl wrapper
  stan::lang::block_var_decl bvar(x);
  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::row_vector_type());
  EXPECT_FALSE(bvar.has_def());

  EXPECT_TRUE(bvar.type().bounds().has_low());
  EXPECT_TRUE(bvar.type().bounds().has_high());
  EXPECT_TRUE(bvar.type().has_def_bounds());

  std::vector<stan::lang::expression> bvar_sizes = bvar.type().size();
  EXPECT_EQ(bvar_sizes.size(), 1);
  EXPECT_TRUE(bvar_sizes.at(0).bare_type().is_int_type());
}

