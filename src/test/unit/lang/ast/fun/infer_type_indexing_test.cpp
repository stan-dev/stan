#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

using std::vector;

TEST(inferTypeIndexing, vector_1d_array_1_idx) {
  stan::lang::vector_type tVec;
  stan::lang::bare_array_type d1(tVec, 1);

  stan::lang::bare_expr_type bet;
  bet = stan::lang::infer_type_indexing(d1, 1);

  EXPECT_TRUE(bet.is_vector_type());
}

TEST(inferTypeIndexing, vector_1d_array_2_idxs) {
  stan::lang::vector_type tVec;
  stan::lang::bare_array_type d1(tVec, 1);

  stan::lang::bare_expr_type bet;
  bet = stan::lang::infer_type_indexing(d1, 2);

  EXPECT_TRUE(bet.is_double_type());
}

TEST(inferTypeIndexing, vector_1d_array_3_idxs) {
  stan::lang::vector_type tVec;
  stan::lang::bare_array_type d1(tVec, 1);

  stan::lang::bare_expr_type bet;
  bet = stan::lang::infer_type_indexing(d1, 3);

  EXPECT_TRUE(bet.is_ill_formed_type());
}

TEST(inferTypeIndexing, vector_5d_array_3_idx) {
  stan::lang::vector_type tVec;
  stan::lang::bare_array_type d1(tVec, 5);

  stan::lang::bare_expr_type bet;
  bet = stan::lang::infer_type_indexing(d1, 3);

  EXPECT_TRUE(bet.is_array_type());
}
