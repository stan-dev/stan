#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>
#include <set>
#include <vector>

using stan::lang::function_signatures;
using stan::lang::expression;
using stan::lang::bare_array_type;
using stan::lang::bare_expr_type;
using stan::lang::ill_formed_type;
using stan::lang::void_type;
using stan::lang::double_type;
using stan::lang::int_type;
using stan::lang::vector_type;
using stan::lang::row_vector_type;
using stan::lang::matrix_type;
using std::vector;

std::vector<bare_expr_type> bare_expr_type_vec() {
  return std::vector<bare_expr_type>();
}

std::vector<bare_expr_type> bare_expr_type_vec(const bare_expr_type& t1) {
  std::vector<bare_expr_type> etv;
  etv.push_back(t1);
  return etv;
}
std::vector<bare_expr_type> bare_expr_type_vec(const bare_expr_type& t1,
                                               const bare_expr_type& t2) {
  std::vector<bare_expr_type> etv;
  etv.push_back(t1);
  etv.push_back(t2);
  return etv;
}
std::vector<bare_expr_type> bare_expr_type_vec(const bare_expr_type& t1,
                                               const bare_expr_type& t2,
                                               const bare_expr_type& t3) {
  std::vector<bare_expr_type> etv;
  etv.push_back(t1);
  etv.push_back(t2);
  etv.push_back(t3);
  return etv;
}

TEST(lang_ast,function_signatures_log_sum_exp_1) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(bare_expr_type(double_type()),
            fs.get_result_type("log_sum_exp",
                               bare_expr_type_vec(bare_array_type(double_type(),1)),
                               error_msgs));
}

TEST(lang_ast,function_signatures_log_sum_exp_2) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(bare_expr_type(double_type()),
            fs.get_result_type("log_sum_exp",
                               bare_expr_type_vec(bare_expr_type(vector_type())),
                               error_msgs));
}

TEST(lang_ast,function_signatures_log_sum_exp_3) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(bare_expr_type(double_type()),
            fs.get_result_type("log_sum_exp",
                               bare_expr_type_vec(bare_expr_type(row_vector_type())),
                               error_msgs));
}

TEST(lang_ast,function_signatures_log_sum_exp_4) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(bare_expr_type(double_type()),
            fs.get_result_type("log_sum_exp",
                               bare_expr_type_vec(bare_expr_type(matrix_type())),
                               error_msgs));
}

TEST(lang_ast,function_signatures_log_sum_exp_binary) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(bare_expr_type(double_type()),
            fs.get_result_type("log_sum_exp",
                               bare_expr_type_vec(bare_expr_type(double_type()),
                                                  bare_expr_type(double_type())),
                               error_msgs));
}

TEST(lang_ast,function_signatures_unary_vectorized_trunc_0) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(bare_expr_type(double_type()),
            fs.get_result_type("trunc",
                               bare_expr_type_vec(double_type()),
                               error_msgs));
}

TEST(lang_ast,function_signatures_unary_vectorized_trunc_1) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(bare_expr_type(bare_array_type(double_type())),
            fs.get_result_type("trunc",
                               bare_expr_type_vec(bare_array_type(double_type())),
                               error_msgs));
}

TEST(lang_ast,function_signatures_unary_vectorized_trunc_8) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(bare_expr_type(bare_array_type(double_type(),8)),
            fs.get_result_type("trunc",
                               bare_expr_type_vec(bare_array_type(double_type(),8)),
                               error_msgs));
}


TEST(lang_ast,function_signatures_multi_normal_0) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(bare_expr_type(double_type()),
            fs.get_result_type("multi_normal_log",
                               bare_expr_type_vec(bare_array_type(vector_type()), bare_array_type(vector_type()), matrix_type()),
                               error_msgs));
}

TEST(lang_ast,function_signatures_multi_normal_cholesky_0) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(bare_expr_type(double_type()),
            fs.get_result_type("multi_normal_cholesky_log",
                               bare_expr_type_vec(bare_array_type(vector_type()), bare_array_type(vector_type()), matrix_type()),
                               error_msgs));
}


TEST(lang_ast,function_signatures_multi_normal_cholesky_1) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(bare_expr_type(double_type()),
            fs.get_result_type("multi_normal_cholesky_log",
                               bare_expr_type_vec(bare_array_type(vector_type()), vector_type(), matrix_type()),
                               error_msgs));
}


TEST(lang_ast, function_signatures_add) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;

  EXPECT_EQ(bare_expr_type(double_type()),
            fs.get_result_type("sqrt", bare_expr_type_vec(bare_expr_type(double_type())),
                               error_msgs));
  EXPECT_EQ(bare_expr_type(),
            fs.get_result_type("foo__", bare_expr_type_vec(), error_msgs));
  EXPECT_EQ(bare_expr_type(),
            fs.get_result_type("foo__", bare_expr_type_vec(bare_expr_type(double_type())), error_msgs));

  // these next two conflict
  fs.add("bar__", bare_expr_type(double_type()), bare_expr_type(int_type()), bare_expr_type(double_type()));
  fs.add("bar__", bare_expr_type(double_type()), bare_expr_type(double_type()), bare_expr_type(int_type()));
  EXPECT_EQ(bare_expr_type(),
            fs.get_result_type("bar__", bare_expr_type_vec(bare_expr_type(int_type()), bare_expr_type(int_type())),
                               error_msgs));

  // after this, should be resolvable
  fs.add("bar__", bare_expr_type(int_type()), bare_expr_type(int_type()), bare_expr_type(int_type()));
  EXPECT_EQ(bare_expr_type(int_type()),
            fs.get_result_type("bar__", bare_expr_type_vec(bare_expr_type(int_type()), bare_expr_type(int_type())),
                               error_msgs));

}

