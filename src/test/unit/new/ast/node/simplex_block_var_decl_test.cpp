#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

TEST(unitVectorBlockVarDecl, createVar1) {
  stan::lang::int_literal int_len(5);
  stan::lang::expression K(int_len);
  stan::lang::unit_vector_block_var_decl x("x", K);

  // check unit_vector_bloc_var_decl
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_vector_type());
  EXPECT_TRUE(is_nil(x.def_));

  // check unit_vector_block_var_type
  stan::lang::expression x_size = x.type_.K_;
  EXPECT_TRUE(x_size.bare_type().is_int_type());

  // check block_var_decl wrapper
  stan::lang::block_var_decl bvar(x);
  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::vector_type());
  EXPECT_FALSE(bvar.has_def());

  EXPECT_FALSE(bvar.type().bounds().has_low());
  EXPECT_FALSE(bvar.type().bounds().has_high());
  EXPECT_FALSE(bvar.type().has_def_bounds());

  std::vector<stan::lang::expression> bvar_sizes = bvar.type().size();
  EXPECT_EQ(bvar_sizes.size(), 1);
  EXPECT_TRUE(bvar_sizes.at(0).bare_type().is_int_type());
}

