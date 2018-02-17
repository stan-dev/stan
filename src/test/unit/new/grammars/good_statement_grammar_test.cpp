#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

TEST(Parser, parse_local_var_assign) {
  std::string input("{\n"
                    "int M;\n"
                    "real x;\n"
                    "x = M;\n"
                    "}\n"
                    "\n");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::statement stmt;
  stmt = parse_statement(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}

TEST(Parser, parse_local_var_assign_2) {
  std::string input("{\n"
                    "int M;\n"
                    "real x[10];\n"
                    "x[1] = M;\n"
                    "}\n"
                    "\n");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::statement stmt;
  stmt = parse_statement(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}

TEST(Parser, parse_local_var_assign_3) {
  std::string input("{\n"
                    "int M[10];\n"
                    "real x[10];\n"
                    "x[1] = M[1];\n"
                    "}\n"
                    "\n");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::statement stmt;
  stmt = parse_statement(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}

TEST(Parser, parse_local_var_assign_4) {
  std::string input("{\n"
                    "  int N;\n"
                    "  int M[N];\n"
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
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}
