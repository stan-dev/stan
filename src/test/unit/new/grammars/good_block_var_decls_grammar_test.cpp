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
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(0 == bvds.size());
}

TEST(Parser, parse_1) {
  std::string input("int<lower=0> x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int< lower>", ss.str());
}

TEST(Parser, parse_2) {
  std::string input("real<lower=0> x;\n"
                    "real<lower=2.1,upper=2.9> y;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  ss << std::endl;
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("real< lower>\n"
            "real< lower, upper>", ss.str());
}

TEST(Parser, parse_3) {
  std::string input("int x;\n"
                    "real y;\n"
                    "real<lower=2.1, upper=2.9> z;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(3 == bvds.size());
}

TEST(Parser, parse_matrix) {
  std::string input("matrix<lower=0>[1,2] my_matrix;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_matrix2) {
  std::string input("matrix<lower=0.0, upper=1.0>[1,2] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_vector) {
  std::string input("vector[5] a;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_vector2) {
  std::string input("vector<lower=0.001>[5] b[10,20,30];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_vector3) {
  std::string input("vector<upper=0.001>[5] c[10,20,30];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_row_vector) {
  std::string input("row_vector[5] a;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_simplex) {
  std::string input("simplex[5] a;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_array_1) {
  std::string input("int N[5];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_array_2) {
  std::string input("int<lower=1> N[5,5];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_1d_array_matrix) {
  std::string input("int x;\nmatrix[2,2] array_of_mat[100];\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
}

TEST(Parser, parse_2d_array_matrix) {
  std::string input("matrix[2,2] d1_array_of_mat[100];\n"
                    "matrix[2,2] d2_array_of_mat[100,100];\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
}
