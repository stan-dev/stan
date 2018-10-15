#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <test/unit/lang/utility.hpp>

TEST(stmtGrammar, fun_as_lhs_sampling) {
  test_parsable("fun_as_lhs_sampling");
  std::string foo = test_parse_msgs("fun_as_lhs_sampling");
  EXPECT_EQ("",foo);
}

TEST(stmtGrammar, fun_as_stmt) {
  test_parsable("fun_as_stmt");
  std::string foo = test_parse_msgs("fun_as_stmt");
  EXPECT_EQ("",foo);
}

