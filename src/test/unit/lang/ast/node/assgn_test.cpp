#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <string>

using stan::lang::assgn;
using stan::lang::bare_expr_type;
using stan::lang::binary_op;
using stan::lang::double_literal;
using stan::lang::double_type;
using stan::lang::expression;
using stan::lang::idx;
using stan::lang::index_op_sliced;
using stan::lang::int_literal;
using stan::lang::int_type;
using stan::lang::omni_idx;
using stan::lang::unary_op;
using stan::lang::uni_idx;
using stan::lang::variable;

using std::vector;

TEST(langAst, assgn) {
  variable v("foo");
  double_type tDouble;
  v.set_type(bare_expr_type(tDouble));
  std::vector<idx> is;
  expression e_int3(int_literal(3));
  uni_idx ui(e_int3);
  idx idx0(ui);
  is.push_back(idx0);
  std::string op("=");
  expression e(int_literal(3));
  assgn a(v, is, op, e);
  // retrieve indexes
  EXPECT_EQ(1, a.idxs_.size());
  // retrieve LHS variable
  EXPECT_EQ(0, a.lhs_var_.type_.num_dims());
  EXPECT_EQ(bare_expr_type(tDouble), a.lhs_var_.type_);
  // retrieve RHS expression
  EXPECT_EQ(0, a.rhs_.bare_type().num_dims());
  int_type tInt;
  EXPECT_EQ(bare_expr_type(tInt), a.rhs_.bare_type());
}

TEST(langAst, assgnSliced) {
  variable v("foo");
  double_type tDouble;
  v.set_type(bare_expr_type(tDouble));
  std::vector<idx> is;
  expression e_int3(int_literal(3));
  uni_idx ui(e_int3);
  idx idx0(ui);
  is.push_back(idx0);
  stan::lang::expression e(v);
  stan::lang::multi_idx i(e);
  idx idx1(i);
  is.push_back(idx1);
  std::string op("=");
  expression e1(double_literal(1.0));
  expression e2(double_literal(2.0));
  std::vector<stan::lang::expression> elements;
  elements.push_back(e1);
  elements.push_back(e1);
  stan::lang::row_vector_expr rv1(elements);
  stan::lang::expression e3 = rv1;
  assgn a(v, is, op, e3);
  // retrieve indexes
  EXPECT_EQ(2, a.idxs_.size());
  EXPECT_TRUE(a.lhs_var_has_sliced_idx());
  // retrieve LHS variable
  //  EXPECT_EQ(0, a.lhs_var_.type_.num_dims());
  //  EXPECT_EQ(bare_expr_type(tDouble), a.lhs_var_.type_);
  // retrieve RHS expression
  //  EXPECT_EQ(0, a.rhs_.bare_type().num_dims());
  //  int_type tInt;
  //  EXPECT_EQ(bare_expr_type(tInt), a.rhs_.bare_type());
}

TEST(langAst, lhsVarOccursOnRhs) {
  variable v("foo");
  v.set_type(double_type());
  std::vector<idx> is;
  expression e_int3(int_literal(3));
  uni_idx ui(e_int3);
  idx idx0(ui);
  is.push_back(idx0);
  std::string op("=");
  expression e(int_literal(3));
  assgn a(v, is, op, e);
  EXPECT_FALSE(a.lhs_var_occurs_on_rhs());

  std::vector<idx> is2;
  assgn a2(v, is2, op, v);
  EXPECT_TRUE(a2.lhs_var_occurs_on_rhs());

  unary_op uo('+', v);
  assgn a3(v, is2, op, uo);
  EXPECT_TRUE(a3.lhs_var_occurs_on_rhs());

  binary_op bo(v, "-", e_int3);
  assgn a4(v, is2, op, bo);
  EXPECT_TRUE(a4.lhs_var_occurs_on_rhs());

  binary_op bo2(e_int3, "*", e_int3);
  assgn a5(v, is2, op, bo2);
  EXPECT_FALSE(a5.lhs_var_occurs_on_rhs());

  binary_op bo3(e_int3, "*", bo);
  assgn a6(v, is2, op, bo3);
  EXPECT_TRUE(a6.lhs_var_occurs_on_rhs());

  index_op_sliced ios(v, is2);
  assgn a7(v, is2, op, ios);
  EXPECT_TRUE(a7.lhs_var_occurs_on_rhs());
}
