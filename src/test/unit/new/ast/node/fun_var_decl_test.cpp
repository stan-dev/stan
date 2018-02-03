#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <iostream>

TEST(arrayFunVarDecl, createVar1) {
  // 1-d array of real
  stan::lang::double_type tDbl;
  stan::lang::bare_array_type batDbl(tDbl);

  stan::lang::fun_var_decl fvar("x", batDbl);

  EXPECT_EQ(fvar.name(), "x");
  EXPECT_EQ(fvar.bare_type(), stan::lang::bare_array_type(stan::lang::double_type()));

  EXPECT_TRUE(fvar.bare_type().is_array_type());
  EXPECT_TRUE(fvar.bare_type().array_contains().is_double_type());
  EXPECT_EQ(fvar.bare_type().array_dims(), 1);
  EXPECT_EQ(fvar.bare_type().num_dims(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, fvar.bare_type());
  EXPECT_EQ("real[ ]", ss.str());
}

TEST(arrayFunVarDecl, createVar2) {
  stan::lang::matrix_type tMat;
  stan::lang::bare_array_type batMat(tMat);

  stan::lang::fun_var_decl fvar("x", batMat);

  EXPECT_EQ(fvar.name(), "x");
  EXPECT_EQ(fvar.bare_type(), stan::lang::bare_array_type(stan::lang::matrix_type()));

  EXPECT_TRUE(fvar.bare_type().is_array_type());
  EXPECT_TRUE(fvar.bare_type().array_contains().is_matrix_type());
  EXPECT_EQ(fvar.bare_type().array_dims(), 1);
  EXPECT_EQ(fvar.bare_type().num_dims(), 3);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, fvar.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());
}

TEST(doubleFunVarDecl, createVar1) {
  stan::lang::double_type tDbl;
  stan::lang::fun_var_decl fvar("x", tDbl);
  EXPECT_EQ(fvar.name(), "x");
  EXPECT_EQ(fvar.bare_type(), stan::lang::double_type());
}

TEST(intLocalVarDecl, createVar1) {
  stan::lang::int_type tInt;
  stan::lang::fun_var_decl fvar("x", tInt);

  EXPECT_EQ(fvar.name(), "x");
  EXPECT_EQ(fvar.bare_type(), stan::lang::int_type());
}

TEST(matrixFunVarDecl, createVar1) {
  stan::lang::matrix_type tMat;
  stan::lang::fun_var_decl fvar("x", tMat);

  EXPECT_EQ(fvar.name(), "x");
  EXPECT_EQ(fvar.bare_type(), stan::lang::matrix_type());
}


TEST(rowVectorFunVarDecl, createVar1) {
  stan::lang::row_vector_type tRowVec;
  stan::lang::fun_var_decl fvar("x", tRowVec);

  EXPECT_EQ(fvar.name(), "x");
  EXPECT_EQ(fvar.bare_type(), stan::lang::row_vector_type());
}

TEST(vectorFunVarDecl, createVar1) {
  stan::lang::vector_type tVec;
  stan::lang::fun_var_decl fvar("x", tVec);

  EXPECT_EQ(fvar.name(), "x");
  EXPECT_EQ(fvar.bare_type(), stan::lang::vector_type());
}

TEST(illFormedFunVarDecl, createVar1) {
  stan::lang::ill_formed_type tIll;
  stan::lang::fun_var_decl fvar("x", tIll);

  EXPECT_EQ(fvar.name(), "x");
  EXPECT_EQ(fvar.bare_type(), stan::lang::ill_formed_type());
}

TEST(illFormedFunVarDecl, createVar2) {
  stan::lang::fun_var_decl fvar;

  EXPECT_EQ(fvar.name(), "");
  EXPECT_EQ(fvar.bare_type(), stan::lang::ill_formed_type());
}

