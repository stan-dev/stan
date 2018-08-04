#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <iostream>

using stan::lang::bare_array_type;
using stan::lang::bare_expr_type;
using stan::lang::double_type;
using stan::lang::int_type;
using stan::lang::ill_formed_type;
using stan::lang::matrix_type;
using stan::lang::row_vector_type;
using stan::lang::vector_type;
using stan::lang::void_type;


TEST(bareExprType, createDefault) {
  bare_expr_type x;
  EXPECT_TRUE(x.is_ill_formed_type());
  EXPECT_EQ(x.num_dims(), 0);
  EXPECT_EQ(x.order_id(),"00_ill_formed_type");
}

TEST(bareExprType, createIllFormed) {
  ill_formed_type tIll;
  EXPECT_EQ(tIll.oid(),"00_ill_formed_type");

  bare_expr_type x(tIll);
  EXPECT_TRUE(x.is_ill_formed_type());
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 0);

  EXPECT_FALSE(x.is_data());
  x.set_is_data();
  EXPECT_FALSE(x.is_data());
  
  bare_expr_type y;
  EXPECT_TRUE(x == y);

  ill_formed_type tIll2;
  bare_expr_type z(tIll2);
  EXPECT_TRUE(x == z);
  EXPECT_TRUE(y == z);
}

TEST(bareExprType, printIllFormed) {
  ill_formed_type tIll;
  EXPECT_EQ(tIll.oid(),"00_ill_formed_type");
  bare_expr_type x(tIll);
  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("ill-formed", ss.str());
}

TEST(bareExprType, createVoid) {
  void_type tVoid;
  EXPECT_EQ(tVoid.oid(),"01_void_type");
  bare_expr_type x(tVoid);
  EXPECT_TRUE(x.is_void_type());
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 0);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("void", ss.str());

  ill_formed_type tIll;
  bare_expr_type z(tIll);
  EXPECT_TRUE(x > z);
  EXPECT_TRUE(x >= z);
}

TEST(bareExprType, createInt) {
  int_type tInt;
  EXPECT_EQ(tInt.oid(),"02_int_type");
  bare_expr_type x(tInt);
  EXPECT_TRUE(x.is_int_type());
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 0);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("int", ss.str());

  EXPECT_FALSE(x.is_data());
  x.set_is_data();
  EXPECT_TRUE(x.is_data());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("data int", ss.str());

  int_type tInt2;
  bare_expr_type y(tInt2);
  EXPECT_TRUE(x == y);
  
  ill_formed_type tIll;
  bare_expr_type z(tIll);
  EXPECT_TRUE(x > z);
  EXPECT_TRUE(x >= z);
  EXPECT_TRUE(z < x);
  EXPECT_TRUE(z <= x);
}

TEST(bareExprType, createInt_1arg) {
  int_type tInt(true);
  EXPECT_EQ(tInt.oid(),"02_int_type");
  bare_expr_type x(tInt);
  EXPECT_TRUE(x.is_int_type());
  EXPECT_TRUE(x.is_data());
  EXPECT_EQ(x.num_dims(), 0);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("data int", ss.str());
}

TEST(bareExprType, createDouble) {
  double_type tDouble;
  EXPECT_EQ(tDouble.oid(),"03_double_type");
  bare_expr_type x(tDouble);
  EXPECT_TRUE(x.is_double_type());
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 0);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("real", ss.str());

  EXPECT_FALSE(x.is_data());
  x.set_is_data();
  EXPECT_TRUE(x.is_data());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("data real", ss.str());
}

TEST(bareExprType, createVector) {
  vector_type tVector;
  EXPECT_EQ(tVector.oid(),"04_vector_type");
  bare_expr_type x(tVector);
  EXPECT_TRUE(x.is_vector_type());
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("vector", ss.str());

  EXPECT_FALSE(x.is_data());
  x.set_is_data();
  EXPECT_TRUE(x.is_data());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("data vector", ss.str());
}

TEST(bareExprType, createRowVector) {
  row_vector_type tRowVector;
  bare_expr_type x(tRowVector);
  EXPECT_TRUE(x.is_row_vector_type());
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("row_vector", ss.str());

  EXPECT_FALSE(x.is_data());
  x.set_is_data();
  EXPECT_TRUE(x.is_data());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("data row_vector", ss.str());
}

TEST(bareExprType, createMatrix) {
  matrix_type tMatrix;
  EXPECT_EQ(tMatrix.oid(),"06_matrix_type");
  bare_expr_type x(tMatrix);
  EXPECT_TRUE(x.is_matrix_type());
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 2);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("matrix", ss.str());

  EXPECT_FALSE(x.is_data());
  x.set_is_data();
  EXPECT_TRUE(x.is_data());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("data matrix", ss.str());
}

TEST(bareExprType, createCopy) {
  int_type tInt;
  bare_expr_type x(tInt);
  bare_expr_type y(x);
  EXPECT_TRUE(y.is_int_type());
  EXPECT_TRUE(x == y);
  EXPECT_EQ(y.num_dims(), 0);
  EXPECT_FALSE(x.is_array_type());
}

TEST(bareExprType, createCopyDataType) {
  bare_expr_type x(int_type(true));
  EXPECT_TRUE(x.is_data());
  bare_expr_type y(x);
  EXPECT_TRUE(y.is_int_type());
  EXPECT_TRUE(y.is_data());
  EXPECT_TRUE(x == y);
  EXPECT_EQ(y.num_dims(), 0);
  EXPECT_FALSE(x.is_array_type());
}

TEST(bareExprType, createCopyDataType2) {
  int_type tInt;
  bare_expr_type x(tInt);
  x.set_is_data();
  EXPECT_TRUE(x.is_data());
  bare_expr_type y(x);
  EXPECT_TRUE(y.is_int_type());
  EXPECT_TRUE(y.is_data());
  EXPECT_TRUE(x == y);
  EXPECT_EQ(y.num_dims(), 0);
  EXPECT_FALSE(x.is_array_type());
}

TEST(bareExprType, createCopyArrayDataType) {
  bare_array_type tDataArrayInt(bare_expr_type(int_type(true)), 2);
  bare_expr_type x(tDataArrayInt);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_TRUE(x.is_data());

  bare_expr_type y(x);
  EXPECT_TRUE(y.is_array_type());
  EXPECT_TRUE(y.array_contains().is_int_type());
  EXPECT_TRUE(y.is_data());

  EXPECT_TRUE(x == y);
  EXPECT_EQ(y.num_dims(), 2);

  std::vector<bare_expr_type> bet_vec;
  bet_vec.push_back(x);
  bare_expr_type z = bet_vec[0];
  EXPECT_TRUE(z.is_array_type());
  EXPECT_TRUE(z.array_contains().is_int_type());
  EXPECT_TRUE(z.is_data());
}

TEST(bareExprType, createArray) {
  bare_array_type d1(bare_expr_type(int_type(false)));
  bare_expr_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);
  EXPECT_FALSE(x.is_data());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("int[ ]", ss.str());
}

TEST(bareExprType, createArrayData) {
  bare_array_type d1(bare_expr_type(int_type(true)));
  bare_expr_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);
  EXPECT_TRUE(x.is_data());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("data int[ ]", ss.str());
}

TEST(bareExprType, createArrayData2) {
  int_type int_type;
  bare_expr_type betInt(int_type);
  betInt.set_is_data();
  EXPECT_TRUE(betInt.is_data());
  bare_array_type d1(betInt);
  bare_expr_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);
  EXPECT_TRUE(x.is_data());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, x);
  EXPECT_EQ("data int[ ]", ss.str());
}

TEST(bareExprType, getArrayElType) {
  int_type tInt;
  bare_array_type d1(tInt);
  bare_expr_type x(d1);
  bare_expr_type y(tInt);
  EXPECT_TRUE(x.is_array_type());
  bare_expr_type z = x.array_element_type();
  EXPECT_TRUE(y == z);
}

TEST(bareExprType, create2DArray) {
  int_type tInt;
  bare_array_type d1(tInt);
  EXPECT_EQ(d1.oid(),"array_02_int_type");
  bare_expr_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  bare_array_type d2(x);
  EXPECT_TRUE(d2.contains() == tInt);
  EXPECT_EQ(d2.dims(), 2);

  bare_expr_type y(d2);
  EXPECT_TRUE(y.is_array_type());
  EXPECT_TRUE(y.array_contains() == tInt);
  EXPECT_EQ(y.array_dims(), 2);
  EXPECT_EQ(y.num_dims(), 2);
  EXPECT_TRUE(y > x);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, y);
  EXPECT_EQ("int[ , ]", ss.str());

  bare_expr_type z = y.array_element_type();
  EXPECT_TRUE(z.is_array_type());
  EXPECT_EQ(z.array_dims(), 1);
  EXPECT_EQ(z.num_dims(), 1);
  EXPECT_TRUE(x == z);

  int_type tInt2;
  bare_expr_type w(tInt2);
  EXPECT_TRUE(x > w);

}

TEST(bareExprType, create2DArrayOfMatrices) {
  matrix_type tMat;
  bare_array_type d1(tMat);
  EXPECT_EQ(d1.oid(),"array_06_matrix_type");
  bare_expr_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 3);
  EXPECT_EQ(x.array_dims(), 1);

  bare_array_type d2(x);
  EXPECT_EQ(d2.oid(),"array_array_06_matrix_type");
  EXPECT_TRUE(d2.contains() == tMat);
  EXPECT_EQ(d2.dims(), 2);

  bare_expr_type y(d2);
  EXPECT_EQ(y.num_dims(), 4);
  EXPECT_EQ(y.array_dims(), 2);
  EXPECT_TRUE(x < y);
  EXPECT_TRUE(x <= y);
  EXPECT_TRUE(x != y);
  
  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, y);
  EXPECT_EQ("matrix[ , ]", ss.str());
}

TEST(bareExprType, create3DArrayOfMatrices) {
  matrix_type tMat;
  bare_array_type d1(tMat, 3);

  EXPECT_EQ(d1.oid(),"array_array_array_06_matrix_type");
  bare_expr_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 5);
  EXPECT_EQ(x.array_dims(), 3);
}

TEST(bareExprType, create1DArrayOfVector) {
  vector_type tVec;
  bare_array_type d1(tVec, 1);
  EXPECT_EQ(d1.oid(),"array_04_vector_type");
  bare_expr_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 2);
  EXPECT_EQ(x.array_dims(), 1);
}

TEST(bareExprType, create0DArrayOfVector) {
  vector_type tVec;
  bare_array_type d1(tVec, 0);
  EXPECT_EQ(d1.oid(),"array_00_ill_formed_type");
  bare_expr_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 0);
  EXPECT_EQ(x.array_dims(), 0);
}

TEST(bareExprType, create10DArrayOfVector) {
  vector_type tVec;
  bare_array_type d1(tVec, 10);
  bare_expr_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 11);
  EXPECT_EQ(x.array_dims(), 10);
}
