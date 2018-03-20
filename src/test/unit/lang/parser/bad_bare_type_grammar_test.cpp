#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <boost/algorithm/string/predicate.hpp>

TEST(Parser, parse_local_unknown) {
  std::string input("unknown");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::bare_expr_type bet;
  bet = parse_bare_type(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER FAILED TO PARSE INPUT"), std::string::npos);
}

TEST(Parser, parse_bare_unclosed_dim) {
  std::string input("int[");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::bare_expr_type bet;
  bet = parse_bare_type(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("<comma to indicate more dimensions or ] "), std::string::npos);
}

TEST(Parser, parse_bare_unclosed_dim_2) {
  std::string input("int[,,,");
  bool pass = false;
  std::stringstream err_msgs;
  stan::lang::bare_expr_type bet;
  bet = parse_bare_type(input, pass, err_msgs);

  EXPECT_NE(err_msgs.str().find("<comma to indicate more dimensions or ] "), std::string::npos);
  EXPECT_FALSE(pass);

}
