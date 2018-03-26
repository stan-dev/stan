#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

TEST(Parser, parse_void) {
  std::string input("functions {\n"
                    "  void fun() {\n"
                    "    print(\"foo\");\n"
                    "  }\n"
                    "}");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fns;
  fns = parse_functions(input, pass, msgs);
  stan::lang::bare_expr_type bet = fns[0].return_type_;
                    
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_void_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("void", ss.str());
}

TEST(Parser, parse_int) {
  std::string input("functions {\n"
                    "  int fun() {\n"
                    "    int foo = 1;\n"
                    "    return foo;\n"
                    "  }\n"
                    "}");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fns;
  fns = parse_functions(input, pass, msgs);
  stan::lang::bare_expr_type bet = fns[0].return_type_;

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_int_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("int", ss.str());
}

TEST(Parser, parse_double) {
  std::string input("functions {\n"
                    "  real fun() {\n"
                    "    real foo = 1.0;\n"
                    "    return foo;\n"
                    "  }\n"
                    "}");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fns;
  fns = parse_functions(input, pass, msgs);
  stan::lang::bare_expr_type bet = fns[0].return_type_;

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_double_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("real", ss.str());
}

TEST(Parser, parse_vector) {
  std::string input("functions {\n"
                    "  vector fun() {\n"
                    "    vector[2] foo;\n"
                    "    return foo;\n"
                    "  }\n"
                    "}");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fns;
  fns = parse_functions(input, pass, msgs);
  stan::lang::bare_expr_type bet = fns[0].return_type_;

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_vector_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("vector", ss.str());
}

TEST(Parser, parse_row_vector) {
  std::string input("functions {\n"
                    "  row_vector fun() {\n"
                    "    row_vector[2] foo;\n"
                    "    return foo;\n"
                    "  }\n"
                    "}");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fns;
  fns = parse_functions(input, pass, msgs);
  stan::lang::bare_expr_type bet = fns[0].return_type_;

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_row_vector_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("row_vector", ss.str());
}

TEST(Parser, parse_matrix) {
  std::string input("functions {\n"
                    "  matrix fun() {\n"
                    "    matrix[2,2] foo;\n"
                    "    return foo;\n"
                    "  }\n"
                    "}");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fns;
  fns = parse_functions(input, pass, msgs);
  stan::lang::bare_expr_type bet = fns[0].return_type_;

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_matrix_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("matrix", ss.str());
}

TEST(Parser, parse_matrix_1d_array) {
  std::string input("functions {\n"
                    "  matrix[ ] fun() {\n"
                    "    matrix[2,2] foo[2];\n"
                    "    return foo;\n"
                    "  }\n"
                    "}");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fns;
  fns = parse_functions(input, pass, msgs);
  stan::lang::bare_expr_type bet = fns[0].return_type_;

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_array_type());
  EXPECT_TRUE(bet.array_element_type().is_matrix_type());
  EXPECT_TRUE(bet.array_contains().is_matrix_type());
  EXPECT_EQ(bet.array_dims(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("matrix[ ]", ss.str());
}

TEST(Parser, parse_matrix_2d_array) {
  std::string input("functions {\n"
                    "  matrix[ , ] fun() {\n"
                    "    matrix[2,2] foo[2,2];\n"
                    "    return foo;\n"
                    "  }\n"
                    "}");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fns;
  fns = parse_functions(input, pass, msgs);
  stan::lang::bare_expr_type bet = fns[0].return_type_;

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_array_type());
  EXPECT_TRUE(bet.array_element_type().is_array_type());
  EXPECT_TRUE(bet.array_contains().is_matrix_type());
  EXPECT_EQ(bet.array_dims(), 2);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("matrix[ , ]", ss.str());
}
