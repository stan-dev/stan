#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

TEST(Parser, parse_indexed_op_lhs) {
  std::string input("{\n"
                    "vector[4] s[5];\n"
                    "vector[3] q;\n"
                    "s[1, :] = q;\n"
                    "}\n"
                    "\n");

  bool pass = false;
  std::stringstream msgs;
  stan::lang::statement stmt;
  stmt = parse_statement(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}
