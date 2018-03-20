#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

TEST(Parser, parse_cholesky_factor_corr_block_type) {
  std::string input("cholesky_factor_corr[3] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("cholesky_factor_corr", ss.str());
}

TEST(Parser, parse_array_of_cholesky_factor_corr_block_type) {
  std::string input("cholesky_factor_corr[3] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("2-dim array of cholesky_factor_corr", ss.str());
}

TEST(Parser, parse_cholesky_factor_cov_block_type_square) {
  std::string input("cholesky_factor_cov[3] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("cholesky_factor_cov", ss.str());
}

TEST(Parser, parse_cholesky_factor_cov_block_type_rect) {
  std::string input("cholesky_factor_cov[3, 4] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("cholesky_factor_cov", ss.str());
}

TEST(Parser, parse_array_of_cholesky_factor_cov_block_type) {
  std::string input("cholesky_factor_cov[3] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("2-dim array of cholesky_factor_cov", ss.str());
}

TEST(Parser, parse_corr_matrix_block_type) {
  std::string input("corr_matrix[3] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("corr_matrix", ss.str());
}

TEST(Parser, parse_array_of_corr_matrix_block_type) {
  std::string input("corr_matrix[3] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("2-dim array of corr_matrix", ss.str());
}

TEST(Parser, parse_cov_matrix_block_type) {
  std::string input("cov_matrix[3] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("cov_matrix", ss.str());
}

TEST(Parser, parse_array_of_cov_matrix_block_type) {
  std::string input("cov_matrix[3] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("2-dim array of cov_matrix", ss.str());
}

TEST(Parser, parse_ordered_block_type) {
  std::string input("ordered[3] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("ordered", ss.str());
}

TEST(Parser, parse_array_of_ordered_block_type) {
  std::string input("ordered[3] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("2-dim array of ordered", ss.str());
}

TEST(Parser, parse_positive_ordered_block_type) {
  std::string input("positive_ordered[3] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("positive_ordered", ss.str());
}

TEST(Parser, parse_array_of_positive_ordered_block_type) {
  std::string input("positive_ordered[3] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("2-dim array of positive_ordered", ss.str());
}

TEST(Parser, parse_simplex_block_type) {
  std::string input("simplex[3] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("simplex", ss.str());
}

TEST(Parser, parse_array_of_simplex_block_type) {
  std::string input("simplex[3] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("2-dim array of simplex", ss.str());
}

TEST(Parser, parse_unit_vector_block_type) {
  std::string input("unit_vector[3] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("unit_vector", ss.str());
}

TEST(Parser, parse_array_of_unit_vector_block_type) {
  std::string input("unit_vector[3] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("2-dim array of unit_vector", ss.str());
}

TEST(Parser, parse_many_types) {
  std::string input("corr_matrix[3] a;\n"
                    "corr_matrix[3] b[5];\n"
                    "corr_matrix[3] c[5, 5];\n"
                    "ordered[3] d;\n"
                    "positive_ordered[3] e[5];\n"
                    "unit_vector[3] f[5, 5];\n");

  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(6 == bvds.size());
  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("corr_matrix", ss.str());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[5].type());
  EXPECT_EQ("2-dim array of unit_vector", ss.str());
}
