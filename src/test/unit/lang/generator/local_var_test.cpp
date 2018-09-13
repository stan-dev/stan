#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/utility.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <boost/variant/polymorphic_get.hpp>


TEST(lang, local_function_body_var_ast) {
  using stan::lang::statements;

  std::string m1("functions {\n"
                 "  void foo() {\n"
                 "    int a;\n"
                 "    real b;\n"
                 "    real c[20,30];\n"
                 "    matrix[40,50] ar_mat[60,70];\n"
                 "    ar_mat[1,1,1,1] = b;\n"
                 "  }\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("function_body_local", m1);
  EXPECT_EQ(1,prog.function_decl_defs_.size());
  statements body = boost::polymorphic_get<statements>(prog.function_decl_defs_[0].body_.statement_);
  EXPECT_EQ(4,body.local_decl_.size());
}

TEST(lang, local_function_body_var_hpp) {
  std::string m1("functions {\n"
                 "  void foo() {\n"
                 "    int a;\n"
                 "    real b;\n"
                 "    real c[20,30];\n"
                 "    matrix[40,50] ar_mat[60,70];\n"
                 "    ar_mat[1,1,1,1] = b;\n"
                 "  }\n"
                 "}\n");
  std::string hpp = model_to_hpp("function_body_local", m1);
  std::string expected("        {\n"
                       "        current_statement_begin__ = 3;\n"
                       "        int a(0);\n"
                       "        (void) a;  // dummy to suppress unused var warning\n"
                       "        stan::math::fill(a, std::numeric_limits<int>::min());\n"
                       "\n"
                       "        current_statement_begin__ = 4;\n"
                       "        local_scalar_t__ b(DUMMY_VAR__);\n"
                       "        (void) b;  // dummy to suppress unused var warning\n"
                       "        stan::math::initialize(b, DUMMY_VAR__);\n"
                       "        stan::math::fill(b, DUMMY_VAR__);\n"
                       "\n"
                       "        current_statement_begin__ = 5;\n"
                       "        validate_non_negative_index(\"c\", \"20\", 20);\n"
                       "        validate_non_negative_index(\"c\", \"30\", 30);\n"
                       "        std::vector<std::vector<local_scalar_t__  >  > c(20, std::vector<local_scalar_t__>(30, local_scalar_t__(DUMMY_VAR__)));\n"
                       "        stan::math::initialize(c, DUMMY_VAR__);\n"
                       "        stan::math::fill(c, DUMMY_VAR__);\n"
                       "\n"
                       "        current_statement_begin__ = 6;\n"
                       "        validate_non_negative_index(\"ar_mat\", \"40\", 40);\n"
                       "        validate_non_negative_index(\"ar_mat\", \"50\", 50);\n"
                       "        validate_non_negative_index(\"ar_mat\", \"60\", 60);\n"
                       "        validate_non_negative_index(\"ar_mat\", \"70\", 70);\n"
                       "        std::vector<std::vector<Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, Eigen::Dynamic>  >  > ar_mat(60, std::vector<Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, Eigen::Dynamic> >(70, Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, Eigen::Dynamic>(40, 50)));\n"
                       "        stan::math::initialize(ar_mat, DUMMY_VAR__);\n"
                       "        stan::math::fill(ar_mat, DUMMY_VAR__);\n");
  EXPECT_EQ(1, count_matches(expected, hpp));
}
