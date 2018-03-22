#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

TEST(Parser, parse_int_literal) {
  std::string input("5");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::expression e;
  e = parse_expression(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(e.bare_type().is_int_type());
}

TEST(Parser, parse_double_literal) {
  std::string input("5.1");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::expression e;
  e = parse_expression(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(e.bare_type().is_double_type());
}

TEST(Parser, parse_array_expr) {
  std::string input("{ 5.1, 6.2, 7.3 }");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::expression e;
  e = parse_expression(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_EQ(input, e.to_string());
}

TEST(Parser, parse_row_vector_expr) {
  std::string input("[ 5.1, 6.2, 7.3 ]");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::expression e;
  e = parse_expression(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_EQ(input, e.to_string());
}

TEST(Parser, parse_matrix_expr) {
  std::string input("[ [ 5.1, 6.2, 7.3 ], [ 5.1, 6.2, 7.3 ] ]");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::expression e;
  e = parse_expression(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_EQ(input, e.to_string());
}

