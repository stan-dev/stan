#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>
#include <set>
#include <vector>

using stan::lang::idx;
using stan::lang::uni_idx;
using stan::lang::omni_idx;
using stan::lang::expression;
using stan::lang::int_literal;
using stan::lang::function_signatures;
using stan::lang::bare_array_type;
using stan::lang::bare_expr_type;
using stan::lang::ill_formed_type;
using stan::lang::void_type;
using stan::lang::double_type;
using stan::lang::int_type;
using stan::lang::vector_type;
using stan::lang::row_vector_type;
using stan::lang::matrix_type;
using std::vector;


TEST(langAst, uniIdx) {
  stan::lang::expression e(stan::lang::int_literal(3));
  stan::lang::uni_idx i(e);
  // test proper type storage and retrieval
  EXPECT_EQ(bare_expr_type(int_type()), i.idx_.bare_type());
  EXPECT_EQ(0, i.idx_.bare_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW((stan::lang::idx)(i));
  stan::lang::idx i2(i);
  EXPECT_EQ(i2.to_string(), "3");
}

TEST(langAst, multiIdx) {
  stan::lang::variable v("foo");
  v.set_type(bare_array_type(int_type(), 1));
  stan::lang::expression e(v);
  stan::lang::multi_idx i(e);
  // test proper type storage and retrieval
  EXPECT_EQ(bare_expr_type(int_type()), i.idxs_.bare_type().innermost_type());
  EXPECT_EQ(1, i.idxs_.bare_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW((stan::lang::idx)(i));
  stan::lang::idx i2(i);
  EXPECT_EQ(i2.to_string(), "foo");
}

TEST(langAst, omniIdx) {
  // nothing to store or retrieve for omni
  EXPECT_NO_THROW(stan::lang::omni_idx());
  // test allow construction
  stan::lang::omni_idx i;
  EXPECT_NO_THROW((stan::lang::idx)(i));
  stan::lang::idx i2(i);
  EXPECT_EQ(i2.to_string(), ":");
}

TEST(langAst, lbIdx) {
  stan::lang::expression e(stan::lang::int_literal(3));
  stan::lang::lb_idx i(e);
  // test proper type storage and retrieval
  EXPECT_EQ(bare_expr_type(int_type()), i.lb_.bare_type());
  EXPECT_EQ(0, i.lb_.bare_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW((stan::lang::idx)(i));
  stan::lang::idx i2(i);
  EXPECT_EQ(i2.to_string(), "3:");
}

TEST(langAst, ubIdx) {
  stan::lang::expression e(stan::lang::int_literal(3));
  stan::lang::ub_idx i(e);
  // test proper type storage and retrieval
  EXPECT_EQ(bare_expr_type(int_type()), i.ub_.bare_type());
  EXPECT_EQ(0, i.ub_.bare_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW((stan::lang::idx)(i));
  stan::lang::idx i2(i);
  EXPECT_EQ(i2.to_string(), ":3");
}

TEST(langAst, lubIdx) {
  stan::lang::expression e1(stan::lang::int_literal(3));
  stan::lang::variable v("foo");
  v.set_type(int_type());
  stan::lang::expression e2(v);
  stan::lang::lub_idx i(e1,e2);
  // test proper type storage and retrieval
  EXPECT_EQ(bare_expr_type(int_type()), i.lb_.bare_type());
  EXPECT_EQ(0, i.lb_.bare_type().num_dims());
  EXPECT_EQ(bare_expr_type(int_type()), i.ub_.bare_type());
  EXPECT_EQ(0, i.ub_.bare_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW((stan::lang::idx)(i));
  stan::lang::idx i2(i);
  EXPECT_EQ(i2.to_string(), "3:foo");
}
