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

TEST(Parser, parse_local_var_assign_for_loop) {
  std::string input("{\n"
                    "  row_vector[2] vs;\n"
                    "  for (v in vs) {\n"
                    "    v = 0;\n"
                    "  }\n"
                    "}\n");

  bool pass = false;
  std::stringstream msgs;
  stan::lang::statement stmt;
  stmt = parse_statement(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}

TEST(Parser, parse_infer_type_1) {
  std::string input("{\n"
                    "  int N;\n"
                    "  int J;\n"
                    "  vector[N] y[J];\n"
                    "  row_vector[N] ry[J];\n"
                    "  for (n in 1:N){\n"
                    "    for (j in 1:J){\n"
                    "      ry[j][n] = y[j][n];\n"
                    "    }\n"
                    "  }\n"
                    "}\n");

  bool pass = false;
  std::stringstream msgs;
  stan::lang::statement stmt;
  stmt = parse_statement(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}


TEST(Parser, parse_infer_type_0) {
  std::string input("{\n"
                    "  int N;\n"
                    "  vector[N] y[N];\n"
                    "  row_vector[N] ry[N];\n"
                    "  ry[1][1] = y[1][1];\n"
                    "}\n");

  bool pass = false;
  std::stringstream msgs;
  stan::lang::statement stmt;
  stmt = parse_statement(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}

TEST(Parser, parse_local_var_def) {
  std::string input("{\n"
                    "int M = 7;\n"
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

TEST(Parser, parse_indexing) {
  std::string input("{\n"
                    "  real y;\n"
                    "  int N;\n"
                    "  int node1[N];\n"
                    "  vector[N] phi;\n"
                    "  y = -0.5 * dot_self(phi[node1] - phi[node1]);\n"
                    "}\n");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::statement stmt;
  stmt = parse_statement(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}

TEST(Parser, keyword_for) {
  std::string input("{\n"
                    "  real force;\n"
                    "  force = force * force;\n"
                    "}\n");
  bool pass = false;
  std::stringstream msgs;
  stan::lang::statement stmt;
  stmt = parse_statement(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}

TEST(Parser, assign_array_expr) {
  std::string input("{\n"
                    "  real d_r1;\n"
                    "  real td_arr_real_d1_2[3] = {d_r1, 2, 3};\n"
                    "}\n");


  bool pass = false;
  std::stringstream msgs;
  stan::lang::statement stmt;
  stmt = parse_statement(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}
