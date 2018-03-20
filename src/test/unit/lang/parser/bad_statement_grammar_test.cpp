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
  stan::lang::statement stmt;
  stmt = parse_statement(input, pass, msgs);
  EXPECT_FALSE(pass);
}

TEST(Parser, parse_assign_index_err_1) {
  std::string input("{\n"
                    "  int N;\n"
                    "  int M;\n"
                    "  real x[N];\n"
                    "  for (i in 1:N) {\n"
                    "    x[i] = M[i];\n"
                    "  }\n"
                    "}\n"
                    "\n");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::statement stmt;
  stmt = parse_statement(input, pass, msgs);
  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("SYNTAX ERROR"), std::string::npos);
  EXPECT_NE(msgs.str().find("indexed expression dimensionality = 0; indexes supplied = 1"), std::string::npos);
}

TEST(Parser, parse_local_var_past_stmt) {
  std::string input("print(\"here\");"
                    "{\n"
                    "  print(\"there\");\n"
                    "  int N;\n"
                    "  }\n"
                    "}\n"
                    "\n");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::statement stmt;
  stmt = parse_statement(input, pass, msgs);
  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER FAILED"), std::string::npos);
}
