#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>
#include <set>
#include <vector>

// was part of test/unit/lang/ast_test.hpp

using stan::lang::bare_array_type;
using stan::lang::bare_expr_type;
using stan::lang::double_type;
using stan::lang::expression;
using stan::lang::ill_formed_type;
using stan::lang::int_type;
using stan::lang::matrix_type;
using stan::lang::row_vector_type;
using stan::lang::variable;
using stan::lang::void_type;
using stan::lang::vector_type;

using std::vector;

TEST(lang_ast, expr_type_num_dims) {
  EXPECT_EQ(0, bare_expr_type().num_dims());
  EXPECT_EQ(2, bare_expr_type(bare_array_type(int_type(), 2)).num_dims());
  EXPECT_EQ(3, bare_expr_type(bare_array_type(vector_type(), 2)).num_dims());
}

TEST(lang_ast, expr_type_is_primitive) {
  EXPECT_TRUE(bare_expr_type(double_type()).is_primitive());
  EXPECT_TRUE(bare_expr_type(int_type()).is_primitive());
  EXPECT_FALSE(bare_expr_type(vector_type()).is_primitive());
  EXPECT_FALSE(bare_expr_type(row_vector_type()).is_primitive());
  EXPECT_FALSE(bare_expr_type(matrix_type()).is_primitive());
  EXPECT_FALSE(bare_expr_type(bare_array_type(int_type(), 2)).is_primitive());
}

TEST(lang_ast, expr_type_is_int_type) {
  EXPECT_FALSE(bare_expr_type(double_type()).is_int_type());
  EXPECT_TRUE(bare_expr_type(int_type()).is_int_type());
  EXPECT_FALSE(bare_expr_type(vector_type()).is_int_type());
  EXPECT_FALSE(bare_expr_type(row_vector_type()).is_int_type());
  EXPECT_FALSE(bare_expr_type(matrix_type()).is_int_type());
  EXPECT_FALSE(bare_expr_type(bare_array_type(int_type(), 2)).is_int_type());
}

TEST(lang_ast, expr_type_is_double_type) {
  EXPECT_TRUE(bare_expr_type(double_type()).is_double_type());
  EXPECT_FALSE(bare_expr_type(int_type()).is_double_type());
  EXPECT_FALSE(bare_expr_type(vector_type()).is_double_type());
  EXPECT_FALSE(bare_expr_type(row_vector_type()).is_double_type());
  EXPECT_FALSE(bare_expr_type(matrix_type()).is_double_type());
  EXPECT_FALSE(bare_expr_type(bare_array_type(int_type(), 2)).is_double_type());
}

TEST(lang_ast,  bare_expr_type_eq) {
  EXPECT_EQ(bare_expr_type(double_type()),  bare_expr_type(double_type()));
  EXPECT_EQ(bare_expr_type(bare_array_type(double_type(), 1)), bare_expr_type(bare_array_type(double_type(), 1)));
  EXPECT_NE(bare_expr_type(int_type()), bare_expr_type(double_type()));
  EXPECT_NE(bare_expr_type(bare_array_type(int_type(), 1)), bare_expr_type(bare_array_type(int_type(), 2)));
  EXPECT_TRUE(bare_expr_type(bare_array_type(int_type(), 1)) != bare_expr_type(bare_array_type(int_type(), 2)));
  EXPECT_FALSE(bare_expr_type(bare_array_type(int_type(), 1)) == bare_expr_type(bare_array_type(int_type(), 2)));
}

TEST(lang_ast, bare_expr_type_compare_ops) {
  EXPECT_TRUE(bare_expr_type(int_type())
              == bare_expr_type(int_type()));
  EXPECT_TRUE(bare_expr_type(int_type())
              != bare_expr_type(double_type()));
  EXPECT_FALSE(bare_expr_type(int_type())
               != bare_expr_type(int_type()));
  EXPECT_TRUE(bare_expr_type(int_type())
              >= bare_expr_type(int_type()));
  EXPECT_TRUE(bare_expr_type(int_type())
              <= bare_expr_type(int_type()));
  EXPECT_FALSE(bare_expr_type(int_type())
               > bare_expr_type(int_type()));
  EXPECT_FALSE(bare_expr_type(int_type())
               < bare_expr_type(int_type()));
  EXPECT_TRUE(bare_expr_type(ill_formed_type())
              < bare_expr_type(int_type()));
  EXPECT_TRUE(bare_expr_type(void_type())
              < bare_expr_type(double_type()));
  EXPECT_TRUE(bare_expr_type(ill_formed_type())
              < bare_expr_type(double_type()));
  EXPECT_TRUE(bare_expr_type(void_type())
              < bare_expr_type(vector_type()));
  EXPECT_TRUE(bare_expr_type(ill_formed_type())
              < bare_expr_type(row_vector_type()));
  EXPECT_TRUE(bare_expr_type(void_type())
              < bare_expr_type(matrix_type()));

  EXPECT_FALSE(bare_expr_type(ill_formed_type())
               < bare_expr_type(ill_formed_type()));
  EXPECT_FALSE(bare_expr_type(void_type())
               < bare_expr_type(void_type()));
  EXPECT_FALSE(bare_expr_type(int_type())
               < bare_expr_type(int_type()));
  EXPECT_FALSE(bare_expr_type(double_type())
               < bare_expr_type(double_type()));
  EXPECT_FALSE(bare_expr_type(vector_type())
               < bare_expr_type(vector_type()));
  EXPECT_FALSE(bare_expr_type(row_vector_type())
               < bare_expr_type(row_vector_type()));
  EXPECT_FALSE(bare_expr_type(matrix_type())
               < bare_expr_type(matrix_type()));

  EXPECT_FALSE(bare_expr_type(ill_formed_type())
               > bare_expr_type(ill_formed_type()));
  EXPECT_FALSE(bare_expr_type(void_type())
               > bare_expr_type(void_type()));
  EXPECT_FALSE(bare_expr_type(int_type())
               > bare_expr_type(int_type()));
  EXPECT_FALSE(bare_expr_type(double_type())
               > bare_expr_type(double_type()));
  EXPECT_FALSE(bare_expr_type(vector_type())
               > bare_expr_type(vector_type()));
  EXPECT_FALSE(bare_expr_type(row_vector_type())
               > bare_expr_type(row_vector_type()));
  EXPECT_FALSE(bare_expr_type(matrix_type())
               > bare_expr_type(matrix_type()));

  EXPECT_FALSE(bare_expr_type(ill_formed_type())
               != bare_expr_type(ill_formed_type()));
  EXPECT_FALSE(bare_expr_type(void_type())
               != bare_expr_type(void_type()));
  EXPECT_FALSE(bare_expr_type(int_type())
               != bare_expr_type(int_type()));
  EXPECT_FALSE(bare_expr_type(double_type())
               != bare_expr_type(double_type()));
  EXPECT_FALSE(bare_expr_type(vector_type())
               != bare_expr_type(vector_type()));
  EXPECT_FALSE(bare_expr_type(row_vector_type())
               != bare_expr_type(row_vector_type()));
  EXPECT_FALSE(bare_expr_type(matrix_type())
               != bare_expr_type(matrix_type()));
}

TEST(lang_ast, bare_expr_type_base) {
  EXPECT_EQ(bare_expr_type(double_type()),
            bare_expr_type(bare_array_type(double_type(), 3)).innermost_type());
  EXPECT_NE(bare_expr_type(double_type()),
            bare_expr_type(bare_array_type(vector_type(), 2)).innermost_type());
}

void testTotalDims(int expected_total_dims,
                   const stan::lang::bare_expr_type& bet,
                   size_t num_dims) {
  variable v("foo");
  if (num_dims == 0)
    v.set_type(bet);
  else
    v.set_type(bare_array_type(bet, num_dims));

  expression e(v);
  EXPECT_EQ(expected_total_dims, e.bare_type().num_dims());
}

TEST(gmAst,expressionTotalDims) {
  testTotalDims(0, double_type(), 0);
  testTotalDims(2, double_type(), 2);
  testTotalDims(0, int_type(), 0);
  testTotalDims(2, int_type(), 2);
  testTotalDims(2, matrix_type(), 0);
  testTotalDims(5, matrix_type(), 3);
  testTotalDims(1, vector_type(), 0);
  testTotalDims(4, vector_type(), 3);
  testTotalDims(1, row_vector_type(), 0);
  testTotalDims(4, row_vector_type(), 3);
}

TEST(langAst, voidType) {
  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, void_type());
  EXPECT_EQ("void", ss.str());
  void_type tVoid;
  bare_expr_type et(tVoid);
  EXPECT_TRUE(et.is_void_type());
}
