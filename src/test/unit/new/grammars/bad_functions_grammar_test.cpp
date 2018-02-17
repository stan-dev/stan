#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

TEST(Parser, parse_empty) {
  std::string input("");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fdds;
  fdds = parse_functions(input, pass, msgs);
  EXPECT_FALSE(pass);
}
