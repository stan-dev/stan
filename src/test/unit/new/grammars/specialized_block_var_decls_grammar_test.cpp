#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

TEST(Parser, parse_cholesky_factor_corr_block_type) {
  std::string input("int K;\n"
                    "cholesky_factor_corr[K] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("cholesky_factor_corr", ss.str());
}

TEST(Parser, parse_array_of_cholesky_factor_corr_block_type) {
  std::string input("int K;\n"
                    "cholesky_factor_corr[K] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("2-dim array of cholesky_factor_corr", ss.str());
}

TEST(Parser, parse_cholesky_factor_cov_block_type_square) {
  std::string input("int K;\n"
                    "cholesky_factor_cov[K] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("cholesky_factor_cov", ss.str());
}

TEST(Parser, parse_cholesky_factor_cov_block_type_rect) {
  std::string input("int M;\n"
                    "int N;\n"
                    "cholesky_factor_cov[M, N] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(3 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[2].type());
  EXPECT_EQ("cholesky_factor_cov", ss.str());
}

TEST(Parser, parse_array_of_cholesky_factor_cov_block_type) {
  std::string input("int K;\n"
                    "cholesky_factor_cov[K] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("2-dim array of cholesky_factor_cov", ss.str());
}

TEST(Parser, parse_corr_matrix_block_type) {
  std::string input("int K;\n"
                    "corr_matrix[K] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("corr_matrix", ss.str());
}

TEST(Parser, parse_array_of_corr_matrix_block_type) {
  std::string input("int K;\n"
                    "corr_matrix[K] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("2-dim array of corr_matrix", ss.str());
}

TEST(Parser, parse_cov_matrix_block_type) {
  std::string input("int K;\n"
                    "cov_matrix[K] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("cov_matrix", ss.str());
}

TEST(Parser, parse_array_of_cov_matrix_block_type) {
  std::string input("int K;\n"
                    "cov_matrix[K] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("2-dim array of cov_matrix", ss.str());
}

TEST(Parser, parse_ordered_block_type) {
  std::string input("int K;\n"
                    "ordered[K] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("ordered", ss.str());
}

TEST(Parser, parse_array_of_ordered_block_type) {
  std::string input("int K;\n"
                    "ordered[K] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("2-dim array of ordered", ss.str());
}

TEST(Parser, parse_positive_ordered_block_type) {
  std::string input("int K;\n"
                    "positive_ordered[K] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("positive_ordered", ss.str());
}

TEST(Parser, parse_array_of_positive_ordered_block_type) {
  std::string input("int K;\n"
                    "positive_ordered[K] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("2-dim array of positive_ordered", ss.str());
}

TEST(Parser, parse_simplex_block_type) {
  std::string input("int K;\n"
                    "simplex[K] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("simplex", ss.str());
}

TEST(Parser, parse_array_of_simplex_block_type) {
  std::string input("int K;\n"
                    "simplex[K] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("2-dim array of simplex", ss.str());
}

TEST(Parser, parse_unit_vector_block_type) {
  std::string input("int K;\n"
                    "unit_vector[K] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("unit_vector", ss.str());
}

TEST(Parser, parse_array_of_unit_vector_block_type) {
  std::string input("int K;\n"
                    "unit_vector[K] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("2-dim array of unit_vector", ss.str());
}

TEST(Parser, parse_many_types) {
  std::string input("int K;\n"
                    "corr_matrix[K] a;\n"
                    "corr_matrix[K] b[5];\n"
                    "corr_matrix[K] c[5, 5];\n"
                    "ordered[K] d;\n"
                    "positive_ordered[K] e[5];\n"
                    "unit_vector[K] f[5, 5];\n");

  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(7 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("corr_matrix", ss.str());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[6].type());
  EXPECT_EQ("2-dim array of unit_vector", ss.str());
}
