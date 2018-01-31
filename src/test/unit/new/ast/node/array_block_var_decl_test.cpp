#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

TEST(arrayBlockVarDecl, createVar1) {
  // 1-d array of real
  stan::lang::double_block_type dbt;
  stan::lang::block_var_type bvtDouble(dbt);
  stan::lang::expression array_len(stan::lang::int_literal(7));
  stan::lang::block_array_type bat(bvtDouble, array_len);
  stan::lang::array_block_var_decl x("x", bat);

  // check block_array_type
  EXPECT_EQ(x.type_.dims(), 1);
  EXPECT_TRUE(x.type_.array_len_.bare_type().is_int_type());
  EXPECT_TRUE(x.type_.element_type_.bare_type().is_double_type());
  
  // // check block_var_decl wrapper
  stan::lang::block_var_decl bvar(x);
  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::bare_array_type(stan::lang::double_type()));
  EXPECT_FALSE(bvar.has_def());

  // get var_decl component
  stan::lang::var_decl vdecl = bvar.var_decl();
  EXPECT_EQ(vdecl.name_, "x");
  EXPECT_TRUE(vdecl.bare_type_.is_array_type());
  EXPECT_TRUE(is_nil(vdecl.def_));

  // check block_array_type component
  EXPECT_TRUE(bvar.type().is_array_type());
  EXPECT_TRUE(bvar.type().array_contains().bare_type().is_double_type());

  EXPECT_FALSE(bvar.type().has_def_bounds());
  EXPECT_FALSE(bvar.type().bounds().has_low());
  EXPECT_FALSE(bvar.type().bounds().has_high());

  std::vector<stan::lang::expression> bvar_sizes = bvar.type().size();
  EXPECT_EQ(bvar_sizes.size(), 1);
  EXPECT_TRUE(bvar_sizes.at(0).bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bvar.bare_type());
  EXPECT_EQ("real[ ]", ss.str());
}

TEST(arrayBlockVarDecl, createVar2) {
  // 1-d array of matrix
  stan::lang::double_literal real_lb(-2.0);
  stan::lang::double_literal real_ub(2.0);
  stan::lang::expression lb(real_lb);
  stan::lang::expression ub(real_ub);
  stan::lang::range m_bounds(lb, ub);
  stan::lang::expression M(stan::lang::int_literal(3));
  stan::lang::expression N(stan::lang::int_literal(4));
  stan::lang::matrix_block_type bvtMatrix(m_bounds, M, N);
  stan::lang::expression array_len(stan::lang::int_literal(7));
  stan::lang::block_array_type bat(bvtMatrix, array_len);
  stan::lang::array_block_var_decl x("x", bat);
  
  // check block_array_type
  EXPECT_TRUE(x.type_.element_type_.bare_type().is_matrix_type());
  EXPECT_EQ(x.type_.dims(), 1);
  EXPECT_TRUE(x.type_.array_len_.bare_type().is_int_type());
  
  // // check block_var_decl wrapper
  stan::lang::block_var_decl bvar(x);
  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::bare_array_type(stan::lang::matrix_type()));
  EXPECT_FALSE(bvar.has_def());

  // get var_decl component
  stan::lang::var_decl vdecl = bvar.var_decl();
  EXPECT_EQ(vdecl.name_, "x");
  EXPECT_TRUE(vdecl.bare_type_.is_array_type());
  EXPECT_TRUE(is_nil(vdecl.def_));

  // check block_array_type component
  EXPECT_TRUE(bvar.type().is_array_type());
  EXPECT_TRUE(bvar.type().array_contains().bare_type().is_matrix_type());

  EXPECT_TRUE(bvar.type().array_contains().has_def_bounds());
  EXPECT_TRUE(bvar.type().array_contains().bounds().has_low());
  EXPECT_TRUE(bvar.type().array_contains().bounds().has_high());

  EXPECT_EQ(bvar.type().num_dims(), 3);
  std::vector<stan::lang::expression> bvar_sizes = bvar.type().size();
  EXPECT_EQ(bvar_sizes.size(), 3);
  EXPECT_TRUE(bvar_sizes.at(0).bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bvar.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());
}

