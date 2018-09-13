#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

using std::vector;

TEST(indexedType, indexed_expr_1d_ar_int_1_idx) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::int_type()));
  stan::lang::expression e1(v1);

  // single non-multi idx
  stan::lang::uni_idx i1;
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(i1));
  std::vector<stan::lang::idx> idxs(1, idx);

  // single index expression over 1-d array returns array element type
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_int_type());
}

TEST(indexedType, indexed_expr_2d_ar_int_1_idx) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::int_type(), 2));
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1;
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(i1));
  std::vector<stan::lang::idx> idxs(1, idx);

  // single index expression over 2-d array returns 1-d array
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_array_type());
  EXPECT_EQ(idx_type.array_dims(), 1);
  EXPECT_TRUE(idx_type.array_contains().is_int_type());
}

TEST(indexedType, indexed_expr_1d_ar_int_1_multi_idx) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::int_type()));
  stan::lang::expression e1(v1);

  // indexed by multi-idx
  stan::lang::multi_idx m1;
  stan::lang::idx idx(m1);
  EXPECT_TRUE(stan::lang::is_multi_index(m1));
  std::vector<stan::lang::idx> idxs(1, idx);

  // single multi-idx index expression over array returns array
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_array_type());
  EXPECT_EQ(idx_type.array_dims(), 1);
  EXPECT_TRUE(idx_type.array_contains().is_int_type());
}

TEST(indexedType, indexed_expr_2d_ar_int_1_multi_idx_1_uni_idx) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::int_type(), 2));
  stan::lang::expression e1(v1);

  stan::lang::multi_idx m1;
  stan::lang::idx idx1(m1);
  stan::lang::uni_idx u1;
  stan::lang::idx idx2(u1);
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx1);
  idxs.push_back(idx2);

  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_EQ(idx_type.array_dims(), 1);
  EXPECT_TRUE(idx_type.array_contains().is_int_type());
}

TEST(indexedType, indexed_expr_1d_ar_double_1_idx) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::double_type()));
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1;
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(i1));
  std::vector<stan::lang::idx> idxs(1, idx);

  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_double_type());
}

TEST(indexedType, indexed_expr_2d_ar_double_1_idx) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::double_type(), 2));
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1;
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(i1));
  std::vector<stan::lang::idx> idxs(1, idx);

  // single index expression over 2-d array returns 1-d array
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_array_type());
  EXPECT_EQ(idx_type.array_dims(), 1);
  EXPECT_TRUE(idx_type.array_contains().is_double_type());
}

TEST(indexedType, indexed_expr_1d_ar_double_2_idxs) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::double_type()));
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1;
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(i1));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx);
  idxs.push_back(idx);

  // too many idxs - should return ill_formed_type
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_ill_formed_type());
}

TEST(indexedType, indexed_expr_3d_ar_double_1_idx) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::double_type(), 3));
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1;
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(i1));
  std::vector<stan::lang::idx> idxs(1, idx);

  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_array_type());
  EXPECT_EQ(idx_type.array_dims(), 2);
  EXPECT_TRUE(idx_type.array_contains().is_double_type());
}

TEST(indexedType, indexed_expr_vector_1_idx) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::vector_type());
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1;
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(i1));
  std::vector<stan::lang::idx> idxs(1, idx);

  // single index expression over vector returns double
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_double_type());
}

TEST(indexedType, indexed_expr_vector_1_multi_idx) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::vector_type());
  stan::lang::expression e1(v1);

  stan::lang::multi_idx m1;
  stan::lang::idx idx(m1);
  EXPECT_TRUE(stan::lang::is_multi_index(idx));
  std::vector<stan::lang::idx> idxs(1, idx);

  // multi index expression over vector returns vector
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_vector_type());
}

TEST(indexedType, indexed_expr_1d_ar_vector_1_idx) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::vector_type()));
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1;
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(i1));
  std::vector<stan::lang::idx> idxs(1, idx);

  // single index expression over vector arr returns vector
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_vector_type());
}

TEST(indexedType, indexed_expr_row_vector_1_idx) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::row_vector_type());
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1;
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(i1));
  std::vector<stan::lang::idx> idxs(1, idx);

  // single index expression over row_vector returns double
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_double_type());
}

TEST(indexedType, indexed_expr_row_vector_1_multi_idx) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::row_vector_type());
  stan::lang::expression e1(v1);

  stan::lang::multi_idx m1;
  stan::lang::idx idx(m1);
  EXPECT_TRUE(stan::lang::is_multi_index(idx));
  std::vector<stan::lang::idx> idxs(1, idx);

  // multi index expression over row_vector returns row_vector
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_row_vector_type());
}

TEST(indexedType, indexed_expr_row_vector_too_many_idxs) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::row_vector_type());
  stan::lang::expression e1(v1);

  stan::lang::multi_idx m1;
  stan::lang::idx idx(m1);
  EXPECT_TRUE(stan::lang::is_multi_index(idx));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx);
  idxs.push_back(idx);

  // too many idxs
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_ill_formed_type());
}

TEST(indexedType, indexed_expr_1d_ar_row_vector_2_multi_idxs) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::row_vector_type()));
  stan::lang::expression e1(v1);

  stan::lang::multi_idx m1;
  stan::lang::idx idx(m1);
  EXPECT_TRUE(stan::lang::is_multi_index(idx));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx);
  idxs.push_back(idx);

  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_array_type());
  EXPECT_EQ(idx_type.array_dims(), 1);
  EXPECT_TRUE(idx_type.array_contains().is_row_vector_type());
}

TEST(indexedType, indexed_expr_matrix_1_idx) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::matrix_type());
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1;
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(idx));
  std::vector<stan::lang::idx> idxs(1, idx);

  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_row_vector_type());
}

TEST(indexedType, indexed_expr_matrix_2_idxs) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::matrix_type());
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1;
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(idx));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx);
  idxs.push_back(idx);

  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_double_type());
}

TEST(indexedType, indexed_expr_1d_arr_matrix_2_idxs) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::matrix_type()));
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1;
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(idx));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx);
  idxs.push_back(idx);

  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_row_vector_type());
}

TEST(indexedType, indexed_expr_1d_arr_matrix_3_idxs) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::matrix_type()));
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1;
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(idx));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx);
  idxs.push_back(idx);
  idxs.push_back(idx);

  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_double_type());
}

TEST(indexedType, indexed_expr_1d_arr_matrix_3_idxs_pos_0_is_multi) {
  stan::lang::multi_idx m1;
  stan::lang::idx idx1(m1);
  EXPECT_TRUE(stan::lang::is_multi_index(idx1));
  stan::lang::uni_idx i1;
  stan::lang::idx idx2(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(idx2));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx1);
  idxs.push_back(idx2);
  idxs.push_back(idx2);

  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::matrix_type()));
  stan::lang::expression e1(v1);

  // m, u, u - reduces array of matrix to array of double
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_EQ(idx_type.num_dims(), 1);
  EXPECT_TRUE(idx_type.is_array_type());
  EXPECT_TRUE(idx_type.array_contains().is_double_type());
}

TEST(indexedType, indexed_expr_1d_arr_matrix_3_idxs_pos_0_1_is_multi) {
  stan::lang::multi_idx m1;
  stan::lang::idx idx1(m1);
  EXPECT_TRUE(stan::lang::is_multi_index(idx1));
  stan::lang::uni_idx i1;
  stan::lang::idx idx2(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(idx2));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx1);
  idxs.push_back(idx1);
  idxs.push_back(idx2);

  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::matrix_type()));
  stan::lang::expression e1(v1);

  // m, m, u - reduces array of matrix to array of vector
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_EQ(idx_type.num_dims(), 2);
  EXPECT_TRUE(idx_type.is_array_type());
  EXPECT_TRUE(idx_type.array_contains().is_vector_type());
}

TEST(indexedType, indexed_expr_1d_arr_matrix_3_idxs_pos_2_is_multi) {
  stan::lang::multi_idx m1;
  stan::lang::idx idx1(m1);
  EXPECT_TRUE(stan::lang::is_multi_index(idx1));
  stan::lang::uni_idx i1(stan::lang::int_literal(7));
  stan::lang::idx idx2(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(idx2));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx2);
  idxs.push_back(idx1);
  idxs.push_back(idx2);

  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::matrix_type()));
  stan::lang::expression e1(v1);

  // u, m, u reduces 1-d arr of matrix to vector
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_vector_type());
}

TEST(indexedType, indexed_expr_1d_arr_matrix_3_idxs_pos_1_2_is_multi) {
  stan::lang::multi_idx m1;
  stan::lang::idx idx1(m1);
  EXPECT_TRUE(stan::lang::is_multi_index(idx1));
  stan::lang::uni_idx i1(stan::lang::int_literal(7));
  stan::lang::idx idx2(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(idx2));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx1);
  idxs.push_back(idx1);
  idxs.push_back(idx2);

  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::matrix_type()));
  stan::lang::expression e1(v1);
  
  // m, m, u reduces 1-d arr of matrix to 1-d arr of vector
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_EQ(idx_type.array_dims(), 1);
  EXPECT_TRUE(idx_type.array_contains().is_vector_type());
}

TEST(indexedType, indexed_expr_2d_arr_matrix_3_idxs) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::matrix_type(), 2));
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1(stan::lang::int_literal(7));
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(idx));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx);
  idxs.push_back(idx);
  idxs.push_back(idx);

  // u, u, u reduces 2-d arr of matrix to row_vec
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_row_vector_type());
}

TEST(indexedType, indexed_expr_1d_arr_matrix_4_idxs) {
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::matrix_type()));
  stan::lang::expression e1(v1);

  stan::lang::uni_idx i1(stan::lang::int_literal(7));
  stan::lang::idx idx(i1);
  EXPECT_FALSE(stan::lang::is_multi_index(idx));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx);
  idxs.push_back(idx);
  idxs.push_back(idx);
  idxs.push_back(idx);

  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_ill_formed_type());
}

TEST(indexedType, prim_type_no_idxs) {
  // note: parser *should* disallow this
  // variable "foo" is int
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::int_type());
  stan::lang::expression e1(v1);

  // empty vec of idx
  std::vector<stan::lang::idx> idxs;

  // single index expression over 1-d array returns array element type
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e1, idxs);
  EXPECT_TRUE(idx_type.is_int_type());
}
