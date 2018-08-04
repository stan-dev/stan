#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>
#include <set>
#include <vector>

using stan::lang::function_decl_def;
using stan::lang::function_decl_defs;
using stan::lang::bare_array_type;
using stan::lang::bare_expr_type;
using stan::lang::double_type;
using stan::lang::int_type;
using stan::lang::no_op_statement;
using stan::lang::statement;
using stan::lang::var_decl;

// tests from old src/test/unit/lang/ast_test.cpp

TEST(langAst,functionDeclDef) {
  function_decl_def fdd(bare_expr_type(int_type()),
                        "foo",
                        std::vector<var_decl>(),
                        statement(no_op_statement()));
  EXPECT_EQ("foo", fdd.name_);
  EXPECT_TRUE(fdd.body_.is_no_op_statement());
  EXPECT_EQ(0U, fdd.arg_decls_.size());
  EXPECT_TRUE(fdd.return_type_.is_int_type());
}

TEST(langAst, functionDeclDefs) {
  function_decl_def fdd1(bare_expr_type(bare_array_type(int_type(),1)),
                         "foo",
                         std::vector<var_decl>(),
                         statement(no_op_statement()));

  var_decl ad("foo", int_type());
  std::vector<var_decl> arg_decls(1, ad);
  function_decl_def fdd2(bare_expr_type(bare_array_type(double_type(), 3)),
                         "bar",
                         arg_decls,
                         statement(no_op_statement()));
  std::vector<function_decl_def> vec_fdds;
  vec_fdds.push_back(fdd1);
  vec_fdds.push_back(fdd2);
  function_decl_defs fdds(vec_fdds);
  EXPECT_EQ(2U, fdds.decl_defs_.size());
}
