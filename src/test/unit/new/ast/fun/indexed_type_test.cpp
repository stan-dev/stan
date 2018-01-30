#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

using std::vector;
TEST(indexedType, indexed_expr_t1) {
  stan::lang::variable v1("foo");
  // multi-idx index expression
  v1.set_type(stan::lang::bare_array_type(stan::lang::int_type()));
  stan::lang::expression e1(v1);
  stan::lang::multi_idx i1(e1);
  stan::lang::idx idx(i1);
  EXPECT_TRUE(stan::lang::is_multi_index(i1));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx);

  stan::lang::bare_expr_type d1 = stan::lang::bare_array_type(stan::lang::int_type());
  stan::lang::bare_expr_type d2 = stan::lang::bare_array_type(d1);
  stan::lang::variable v2("bar");
  v2.set_type(d2);
  stan::lang::expression e2(v2);
  
  // single multi-idx index expression over array returns array
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e2, idxs);
  EXPECT_TRUE(d2 == idx_type);
}

TEST(indexedType, indexed_expr_t2) {
  stan::lang::variable v1("foo");
  // multi-idx index expression
  v1.set_type(stan::lang::bare_array_type(stan::lang::int_type()));
  stan::lang::expression e1(v1);
  stan::lang::multi_idx i1(e1);
  stan::lang::idx idx(i1);
  EXPECT_TRUE(stan::lang::is_multi_index(i1));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx);
  idxs.push_back(idx);

  stan::lang::bare_expr_type d1 = stan::lang::bare_array_type(stan::lang::int_type());
  stan::lang::bare_expr_type d2 = stan::lang::bare_array_type(d1);
  stan::lang::variable v2("bar");
  v2.set_type(d2);
  stan::lang::expression e2(v2);
  
  // two multi-idx index expressions over 2D array returns array
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e2, idxs);
  EXPECT_TRUE(d2 == idx_type);
}

TEST(indexedType, indexed_expr_t3) {
  // multi-idx index expression
  stan::lang::variable v1("a");
  v1.set_type(stan::lang::bare_array_type(stan::lang::int_type()));
  stan::lang::expression e1(v1);
  stan::lang::multi_idx i1(e1);
  stan::lang::idx m_idx(i1);
  EXPECT_TRUE(stan::lang::is_multi_index(m_idx));

  // uni-idx index expression
  stan::lang::variable v2("b");
  v2.set_type(stan::lang::int_type());
  stan::lang::expression e2(v2);
  stan::lang::uni_idx i2(e2);
  stan::lang::idx u_idx(i2);
  EXPECT_FALSE(stan::lang::is_multi_index(u_idx));

  std::vector<stan::lang::idx> idxs;
  idxs.push_back(m_idx);
  idxs.push_back(u_idx);

  stan::lang::bare_expr_type d1 = stan::lang::bare_array_type(stan::lang::int_type());
  stan::lang::bare_expr_type d2 = stan::lang::bare_array_type(d1);
  stan::lang::variable v3("bar");
  v3.set_type(d2);
  stan::lang::expression e3(v3);
  
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e3, idxs);
  EXPECT_TRUE(d1 == idx_type);
}

TEST(indexedType, indexed_expr_t4) {
  // uni-idx index expression
  stan::lang::variable v2("b");
  v2.set_type(stan::lang::int_type());
  stan::lang::expression e2(v2);
  stan::lang::uni_idx i2(e2);
  stan::lang::idx u_idx(i2);
  EXPECT_FALSE(stan::lang::is_multi_index(u_idx));

  std::vector<stan::lang::idx> idxs;
  idxs.push_back(u_idx);
  idxs.push_back(u_idx);

  stan::lang::bare_expr_type d1 = stan::lang::bare_array_type(stan::lang::int_type());
  stan::lang::bare_expr_type d2 = stan::lang::bare_array_type(d1);
  stan::lang::variable v3("bar");
  v3.set_type(d2);
  stan::lang::expression e3(v3);
  
  stan::lang::bare_expr_type idx_type = stan::lang::indexed_type(e3, idxs);
  EXPECT_TRUE(idx_type.is_int_type());
}
