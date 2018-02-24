#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>

#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

using std::vector;

TEST(generateMemberVarInits, t1) {

  stan::lang::double_block_type dbt;
  stan::lang::block_var_type bvtDouble(dbt);
  stan::lang::expression array_len(stan::lang::int_literal(7));
  stan::lang::block_array_type bat(bvtDouble, array_len);
  stan::lang::block_var_decl bvar("x", bat);
  std::vector<block_var_decl> bvds(1, bvar);

  EXPECT_TRUE(true);
}
