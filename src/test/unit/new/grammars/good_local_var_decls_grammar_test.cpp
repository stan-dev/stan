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
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(0 == lvds.size());
}

TEST(Parser, parse_1) {
  std::string input("int N;\n"
                    "int M;\n"
                    "real x;\n"
                    "real y;\n"
                    "real z;\n"
                    "\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(5 == lvds.size());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, lvds[0].bare_type());
  EXPECT_EQ("int", ss.str());
}

TEST(Parser, parse_matrix) {
  std::string input("matrix[2,2] my_matrix;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;

  lvds = parse_local_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == lvds.size());
}

TEST(Parser, parse_vector) {
  std::string input("vector[5] a;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;

  lvds = parse_local_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == lvds.size());
}

TEST(Parser, parse_row_vector) {
  std::string input("row_vector[5] a;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;

  lvds = parse_local_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == lvds.size());
}

TEST(Parser, parse_row_vector2) {
  std::string input("int N;\n"
                    "row_vector[N] a;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == lvds.size());
}

TEST(Parser, parse_array_1) {
  std::string input("int N[5];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == lvds.size());
}

TEST(Parser, parse_1d_array_matrix) {
  std::string input("int x;\n"
                    "matrix[2,2] array_of_mat[100];\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == lvds.size());
}

TEST(Parser, parse_2d_array_matrix) {
  std::string input("matrix[2,2] d1_array_of_mat[100];\n"
                    "matrix[2,2] d2_array_of_mat[100,100];\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == lvds.size());
}

TEST(Parser, parse_2d_array_matrix_2) {
  std::string input("int N;\n"
                    "int M;\n"
                    "int I;\n"
                    "int J;\n"
                    "matrix[M,N] d1_array_of_mat[I];\n"
                    "matrix[M,N] d2_array_of_mat[I, J];\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(6 == lvds.size());
}
