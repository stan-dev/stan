#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <iostream>

using stan::lang::block_var_type;
using stan::lang::block_array_type;
using stan::lang::cholesky_corr_block_type;
using stan::lang::cholesky_factor_block_type;
using stan::lang::corr_matrix_block_type;
using stan::lang::cov_matrix_block_type;
using stan::lang::double_block_type;
using stan::lang::ill_formed_type;
using stan::lang::int_block_type;
using stan::lang::matrix_block_type;
using stan::lang::ordered_block_type;
using stan::lang::positive_ordered_block_type;
using stan::lang::row_vector_block_type;
using stan::lang::simplex_block_type;
using stan::lang::unit_vector_block_type;
using stan::lang::vector_block_type;

using stan::lang::expression;
using stan::lang::range;

TEST(blockVarType, createDefault) {
  block_var_type x;
  EXPECT_EQ(x.num_dims(), 0);
}

TEST(blockVarType, createIllFormed) {
  ill_formed_type tIll;
  block_var_type x(tIll);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 0);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 0);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("ill formed", ss.str());
}

TEST(blockVarType, createInt) {
  range r1;
  int_block_type tInt(r1);
  block_var_type x(tInt);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 0);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 0);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("int", ss.str());
}

TEST(blockVarType, createDouble) {
  double_block_type tDouble;
  block_var_type x(tDouble);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 0);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 0);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("real", ss.str());
}

TEST(blockVarType, createDoubleBounded) {
  range r1;
  double_block_type tDouble(r1);
  block_var_type x(tDouble);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 0);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 0);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("real", ss.str());
}

TEST(blockVarType, createVector) {
  vector_block_type tVector;
  block_var_type x(tVector);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector", ss.str());
}

TEST(blockVarType, createRowVector) {
  row_vector_block_type tRowVector;
  block_var_type x(tRowVector);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("row vector", ss.str());
}

TEST(blockVarType, createMatrix) {
  matrix_block_type tMatrix;
  block_var_type x(tMatrix);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 2);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(blockVarType, createMatrixSized) {
  range r1;
  expression e1;
  matrix_block_type tMatrix(r1, e1, e1);
  block_var_type x(tMatrix);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 2);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(blockVarType, createVectorSized) {
  range r1;
  expression e1;
  vector_block_type tVector(r1, e1);
  block_var_type x(tVector);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector", ss.str());
}

TEST(blockVarType, createRowVectorSized) {
  range r1;
  expression e1;
  row_vector_block_type tRowVector(r1, e1);
  block_var_type x(tRowVector);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("row vector", ss.str());
}

TEST(blockVarType, createCopy) {
  int_block_type tInt;
  block_var_type x(tInt);
  block_var_type y(x);
  //  EXPECT_TRUE(x == y);
  EXPECT_EQ(y.num_dims(), 0);
  EXPECT_FALSE(y.is_array_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, y.bare_type());
  EXPECT_EQ("int", ss.str());
}

TEST(blockVarType, createArray) {
  int_block_type tInt;
  expression e1;
  block_array_type d1(tInt,e1);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);
  expression array_len = x.array_len();

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("int[ ]", ss.str());
}

TEST(blockVarType, getArrayElType) {
  int_block_type tInt;
  expression e1;
  block_array_type d1(tInt,e1);
  block_var_type x(d1);
  block_var_type y(tInt);
  EXPECT_TRUE(x.is_array_type());

  block_var_type z = x.array_element_type();
  EXPECT_FALSE(z.is_array_type());
  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, z.bare_type());
  EXPECT_EQ("int", ss.str());

}

TEST(blockVarType, create2DArray) {
  int_block_type tInt;
  expression e1;
  block_array_type d1(tInt,e1);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  expression e2;
  block_array_type d2(x,e2);
  EXPECT_EQ(d2.dims(), 2);

  block_var_type y(d2);
  EXPECT_TRUE(y.is_array_type());
  EXPECT_EQ(y.array_dims(), 2);
  EXPECT_EQ(y.num_dims(), 2);

  std::vector<expression> sizes = y.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, y.bare_type());
  EXPECT_EQ("int[ , ]", ss.str());

  block_var_type z = y.array_element_type();
  EXPECT_TRUE(z.is_array_type());
  EXPECT_EQ(z.array_dims(), 1);
  EXPECT_EQ(z.num_dims(), 1);
}

TEST(blockVarType, create2DArrayOfMatrices) {
  range r1;
  expression e1;
  expression e2;
  expression e3;
  expression e4;

  matrix_block_type tMat(r1, e1, e2);
  block_array_type d1(tMat,e3);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 3);
  EXPECT_EQ(x.array_dims(), 1);

  block_array_type d2(x,e4);
  EXPECT_EQ(d2.dims(), 2);

  block_var_type y(d2);
  EXPECT_EQ(y.num_dims(), 4);
  EXPECT_EQ(y.array_dims(), 2);

  std::vector<expression> sizes = y.size();
  EXPECT_EQ(sizes.size(), 4);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, y.bare_type());
  EXPECT_EQ("matrix[ , ]", ss.str());

}

TEST(blockVarType, createArrayInt) {
  range r1;
  expression e1;

  int_block_type tInt(r1);

  std::vector<expression> dims;
  dims.push_back(e1);

  block_array_type d1(tInt,dims);
  block_var_type y(d1);
  EXPECT_TRUE(y.is_array_type());
  EXPECT_TRUE(y.array_contains().bare_type().is_int_type());
  EXPECT_EQ(y.array_dims(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, y.bare_type());
  EXPECT_EQ("int[ ]", ss.str());
}

TEST(blockVarType, create2DArrayInt) {
  range r1;
  expression e1;
  expression e2;

  int_block_type tInt(r1);

  std::vector<expression> dims;
  dims.push_back(e1);
  dims.push_back(e2);

  block_array_type d2(tInt,dims);
  block_var_type y(d2);
  EXPECT_TRUE(y.is_array_type());
  EXPECT_TRUE(y.array_contains().bare_type().is_int_type());
  EXPECT_EQ(y.array_dims(), 2);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, y.bare_type());
  EXPECT_EQ("int[ , ]", ss.str());
}

TEST(blockVarType, create4DArrayInt) {
  range r1;
  expression e1;
  expression e2;
  expression e3;
  expression e4;

  int_block_type tInt(r1);

  std::vector<expression> dims;
  dims.push_back(e1);
  dims.push_back(e2);
  dims.push_back(e3);
  dims.push_back(e4);

  block_array_type d4(tInt,dims);
  block_var_type y(d4);
  EXPECT_TRUE(y.is_array_type());
  EXPECT_TRUE(y.array_contains().bare_type().is_int_type());
  EXPECT_EQ(y.array_dims(), 4);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, y.bare_type());
  EXPECT_EQ("int[ , , , ]", ss.str());
}
