#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <iostream>

using stan::lang::local_array_type;
using stan::lang::local_var_type;
using stan::lang::double_type;
using stan::lang::int_type;
using stan::lang::ill_formed_type;
using stan::lang::matrix_local_type;
using stan::lang::row_vector_local_type;
using stan::lang::vector_local_type;
using stan::lang::expression;

TEST(localVarType, createDefault) {
  local_var_type x;
  EXPECT_EQ(x.num_dims(), 0);
}

TEST(localVarType, createIllFormed) {
  ill_formed_type tIll;
  local_var_type x(tIll);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 0);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 0);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("ill formed", ss.str());
}

TEST(localVarType, createInt) {
  int_type tInt;
  local_var_type x(tInt);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 0);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 0);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("int", ss.str());
}

TEST(localVarType, createDouble) {
  double_type tDouble;
  local_var_type x(tDouble);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 0);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 0);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("real", ss.str());
}

TEST(localVarType, createVector) {
  vector_local_type tVector;
  local_var_type x(tVector);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector", ss.str());
}

TEST(localVarType, createRowVector) {
  row_vector_local_type tRowVector;
  local_var_type x(tRowVector);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("row vector", ss.str());
}

TEST(localVarType, createMatrix) {
  matrix_local_type tMatrix;
  local_var_type x(tMatrix);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 2);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(localVarType, createMatrixSized) {
  expression e;
  matrix_local_type tMatrix(e, e);
  local_var_type x(tMatrix);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 2);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(localVarType, createVectorSized) {
  expression e;
  vector_local_type tVector(e);
  local_var_type x(tVector);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector", ss.str());
}

TEST(localVarType, createRowVectorSized) {
  expression e;
  row_vector_local_type tRowVector(e);
  local_var_type x(tRowVector);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("row vector", ss.str());
}

TEST(localVarType, createCopy) {
  int_type tInt;
  local_var_type x(tInt);
  local_var_type y(x);
  //  EXPECT_TRUE(x == y);
  EXPECT_EQ(y.num_dims(), 0);
  EXPECT_FALSE(y.is_array_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, y.bare_type());
  EXPECT_EQ("int", ss.str());
}

TEST(localVarType, createArray) {
  int_type tInt;
  expression e;
  local_array_type d1(tInt,e);
  local_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);
  expression array_len = x.array_len();

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("int[ ]", ss.str());
}

TEST(localVarType, getArrayElType) {
  int_type tInt;
  expression e;
  local_array_type d1(tInt,e);
  local_var_type x(d1);
  local_var_type y(tInt);
  EXPECT_TRUE(x.is_array_type());

  local_var_type z = x.array_element_type();
  EXPECT_FALSE(z.is_array_type());
  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, z.bare_type());
  EXPECT_EQ("int", ss.str());

}

TEST(localVarType, create2DArray) {
  int_type tInt;
  expression e1;
  local_array_type d1(tInt,e1);
  local_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  expression e2;
  local_array_type d2(x,e2);
  EXPECT_EQ(d2.dims(), 2);

  local_var_type y(d2);
  EXPECT_TRUE(y.is_array_type());
  EXPECT_EQ(y.array_dims(), 2);
  EXPECT_EQ(y.num_dims(), 2);

  std::vector<expression> sizes = y.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, y.bare_type());
  EXPECT_EQ("int[ , ]", ss.str());

  local_var_type z = y.array_element_type();
  EXPECT_TRUE(z.is_array_type());
  EXPECT_EQ(z.array_dims(), 1);
  EXPECT_EQ(z.num_dims(), 1);
}

TEST(localVarType, create2DArrayOfMatrices) {
  expression e1;
  expression e2;
  expression e3;
  expression e4;

  matrix_local_type tMat(e1, e2);
  local_array_type d1(tMat,e3);
  local_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 3);
  EXPECT_EQ(x.array_dims(), 1);

  local_array_type d2(x,e4);
  EXPECT_EQ(d2.dims(), 2);

  local_var_type y(d2);
  EXPECT_EQ(y.num_dims(), 4);
  EXPECT_EQ(y.array_dims(), 2);

  std::vector<expression> sizes = y.size();
  EXPECT_EQ(sizes.size(), 4);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, y.bare_type());
  EXPECT_EQ("matrix[ , ]", ss.str());

}

TEST(localVarType, create4DArrayInt) {
  expression e1;
  expression e2;
  expression e3;
  expression e4;

  int_type tInt;

  std::vector<expression> dims;
  dims.push_back(e1);
  dims.push_back(e2);
  dims.push_back(e3);
  dims.push_back(e4);

  local_array_type d4(tInt,dims);
  local_var_type y(d4);
  EXPECT_TRUE(y.is_array_type());
  EXPECT_TRUE(y.array_contains().bare_type().is_int_type());
  EXPECT_EQ(y.array_dims(), 4);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, y.bare_type());
  EXPECT_EQ("int[ , , , ]", ss.str());
}
