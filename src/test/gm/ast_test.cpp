#include <gtest/gtest.h>
#include <sstream>
#include <cmath>
#include <vector>
#include "stan/gm/ast_def.cpp"

using stan::gm::function_signatures;
using stan::gm::expr_type;
using stan::gm::DOUBLE_T;
using stan::gm::INT_T;
using stan::gm::VECTOR_T;
using stan::gm::ROW_VECTOR_T;
using stan::gm::MATRIX_T;

TEST(gm_ast,expr_type_num_dims) {
  EXPECT_EQ(0U,expr_type().num_dims());
  EXPECT_EQ(2U,expr_type(INT_T,2U).num_dims());
  EXPECT_EQ(2U,expr_type(VECTOR_T,2U).num_dims());
}
TEST(gm_ast,expr_type_is_primitive) {
  EXPECT_TRUE(expr_type(DOUBLE_T).is_primitive());
  EXPECT_TRUE(expr_type(INT_T).is_primitive());
  EXPECT_FALSE(expr_type(VECTOR_T).is_primitive());
  EXPECT_FALSE(expr_type(ROW_VECTOR_T).is_primitive());
  EXPECT_FALSE(expr_type(MATRIX_T).is_primitive());
  EXPECT_FALSE(expr_type(INT_T,2U).is_primitive());
}
TEST(gm_ast,expr_type_is_primitive_int) {
  EXPECT_FALSE(expr_type(DOUBLE_T).is_primitive_int());
  EXPECT_TRUE(expr_type(INT_T).is_primitive_int());
  EXPECT_FALSE(expr_type(VECTOR_T).is_primitive_int());
  EXPECT_FALSE(expr_type(ROW_VECTOR_T).is_primitive_int());
  EXPECT_FALSE(expr_type(MATRIX_T).is_primitive_int());
  EXPECT_FALSE(expr_type(INT_T,2U).is_primitive_int());
}
TEST(gm_ast,expr_type_is_primitive_double) {
  EXPECT_TRUE(expr_type(DOUBLE_T).is_primitive_double());
  EXPECT_FALSE(expr_type(INT_T).is_primitive_double());
  EXPECT_FALSE(expr_type(VECTOR_T).is_primitive_double());
  EXPECT_FALSE(expr_type(ROW_VECTOR_T).is_primitive_double());
  EXPECT_FALSE(expr_type(MATRIX_T).is_primitive_double());
  EXPECT_FALSE(expr_type(INT_T,2U).is_primitive_double());
}
TEST(gm_ast,expr_type_eq) {
  EXPECT_EQ(expr_type(DOUBLE_T),expr_type(DOUBLE_T));
  EXPECT_EQ(expr_type(DOUBLE_T,1U),expr_type(DOUBLE_T,1U));
  EXPECT_NE(expr_type(INT_T), expr_type(DOUBLE_T));
  EXPECT_NE(expr_type(INT_T,1), expr_type(INT_T,2));
}
TEST(gm_ast,expr_type_type) {
  EXPECT_EQ(DOUBLE_T,expr_type(DOUBLE_T).type());
  EXPECT_EQ(DOUBLE_T,expr_type(DOUBLE_T,3U).type());
  EXPECT_NE(DOUBLE_T,expr_type(INT_T).type());
  EXPECT_NE(DOUBLE_T,expr_type(VECTOR_T,2U).type());
}

std::vector<expr_type> expr_type_vec() {
  return std::vector<expr_type>();
}
std::vector<expr_type> expr_type_vec(const expr_type& t1) {
  std::vector<expr_type> etv;
  etv.push_back(t1);
  return etv;
}
std::vector<expr_type> expr_type_vec(const expr_type& t1,
                                     const expr_type& t2) {
  std::vector<expr_type> etv;
  etv.push_back(t1);
  etv.push_back(t2);
  return etv;
}
std::vector<expr_type> expr_type_vec(const expr_type& t1,
                                     const expr_type& t2,
                                     const expr_type& t3) {
  std::vector<expr_type> etv;
  etv.push_back(t1);
  etv.push_back(t2);
  etv.push_back(t3);
  return etv;
}

TEST(gm_ast,function_signatures_log_sum_exp_1) {
  stan::gm::function_signatures& fs = stan::gm::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(expr_type(DOUBLE_T),
            fs.get_result_type("log_sum_exp",
                               expr_type_vec(expr_type(DOUBLE_T,1U)),
                               error_msgs));
}

TEST(gm_ast,function_signatures_log_sum_exp_2) {
  stan::gm::function_signatures& fs = stan::gm::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(expr_type(DOUBLE_T),
            fs.get_result_type("log_sum_exp",
                               expr_type_vec(expr_type(DOUBLE_T),
                                             expr_type(DOUBLE_T)),
                               error_msgs));
}

TEST(gm_ast,function_signatures_add) {
  stan::gm::function_signatures& fs = stan::gm::function_signatures::instance();
  std::stringstream error_msgs;

  EXPECT_EQ(expr_type(DOUBLE_T), 
            fs.get_result_type("sqrt",expr_type_vec(expr_type(DOUBLE_T)),
                               error_msgs));
  EXPECT_EQ(expr_type(), fs.get_result_type("foo__",expr_type_vec(),error_msgs));
  EXPECT_EQ(expr_type(), fs.get_result_type("foo__",expr_type_vec(expr_type(DOUBLE_T)),error_msgs));

  // these next two conflict
  fs.add("bar__",expr_type(DOUBLE_T),expr_type(INT_T),expr_type(DOUBLE_T));
  fs.add("bar__",expr_type(DOUBLE_T),expr_type(DOUBLE_T),expr_type(INT_T));
  EXPECT_EQ(expr_type(), 
            fs.get_result_type("bar__",expr_type_vec(expr_type(INT_T),expr_type(INT_T)),
                               error_msgs));

  // after this, should be resolvable
  fs.add("bar__",expr_type(INT_T), expr_type(INT_T), expr_type(INT_T));
  EXPECT_EQ(expr_type(INT_T), 
            fs.get_result_type("bar__",expr_type_vec(INT_T,INT_T),
                               error_msgs)); // expr_type(INT_T),expr_type(INT_T))));
  
}

