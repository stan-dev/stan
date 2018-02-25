#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

TEST(Parser, parse_void) {
  std::string input("void");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::bare_expr_type bet;
  bet = parse_bare_type(input, pass, msgs);
  std::cout << msgs.str() << std::endl;

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_void_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("void", ss.str());
}

TEST(Parser, parse_int) {
  std::string input("int");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::bare_expr_type bet;
  bet = parse_bare_type(input, pass, msgs);
  std::cout << msgs.str() << std::endl;

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_int_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("int", ss.str());
}

TEST(Parser, parse_double) {
  std::string input("real");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::bare_expr_type bet;
  bet = parse_bare_type(input, pass, msgs);
  std::cout << msgs.str() << std::endl;

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_double_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("real", ss.str());
}

TEST(Parser, parse_vector) {
  std::string input("vector");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::bare_expr_type bet;
  bet = parse_bare_type(input, pass, msgs);
  std::cout << msgs.str() << std::endl;

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_vector_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("vector", ss.str());
}

TEST(Parser, parse_row_vector) {
  std::string input("row_vector");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::bare_expr_type bet;
  bet = parse_bare_type(input, pass, msgs);
  std::cout << msgs.str() << std::endl;

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_row_vector_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("row vector", ss.str());
}

TEST(Parser, parse_matrix) {
  std::string input("matrix");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::bare_expr_type bet;
  bet = parse_bare_type(input, pass, msgs);
  std::cout << msgs.str() << std::endl;

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(bet.is_matrix_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bet);
  EXPECT_EQ("matrix", ss.str());
}

TEST(Parser, parse_matrix_1d_array) {
  std::string input("matrix[]");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::bare_expr_type bet;
  bet = parse_bare_type(input, pass, msgs);
  std::cout << msgs.str() << std::endl;

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
  std::string input("matrix[,]");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::bare_expr_type bet;
  bet = parse_bare_type(input, pass, msgs);
  std::cout << msgs.str() << std::endl;

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
