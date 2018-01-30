#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

TEST(choleskyFactorBlockVarDecl, createVar1) {
  stan::lang::expression M(stan::lang::int_literal(3));
  stan::lang::expression N(stan::lang::int_literal(4));
  stan::lang::cholesky_factor_block_var_decl x("x", M, N);

  // check cholesky_factor_bloc_var_decl
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_matrix_type());
  EXPECT_TRUE(is_nil(x.def_));

  // check cholesky_factor_block_var_type
  stan::lang::expression m_size = x.type_.M_;
  EXPECT_TRUE(m_size.bare_type().is_int_type());
  stan::lang::expression n_size = x.type_.N_;
  EXPECT_TRUE(m_size.bare_type().is_int_type());

  // check block_var_decl wrapper
  stan::lang::block_var_decl bvar(x);
  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::matrix_type());
  EXPECT_FALSE(bvar.has_def());

  EXPECT_FALSE(bvar.type().has_def_bounds());

  std::vector<stan::lang::expression> bvar_sizes = bvar.type().size();
  EXPECT_EQ(bvar_sizes.size(), 2);
  EXPECT_TRUE(bvar_sizes.at(0).bare_type().is_int_type());
}

