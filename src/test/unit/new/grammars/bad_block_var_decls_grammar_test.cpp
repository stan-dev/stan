#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <boost/algorithm/string/predicate.hpp>

TEST(Parser, parse_unknown) {
  std::string input("unknown x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER FAILED TO PARSE INPUT"), std::string::npos);
}

TEST(Parser, parse_unfinished_1) {
  std::string input("int;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  
  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: <identifier>"), std::string::npos);
}

TEST(Parser, parse_unfinished_2) {
  std::string input("int x");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  
  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED"), std::string::npos);
  EXPECT_NE(msgs.str().find("\";\""), std::string::npos);
}

TEST(Parser, parse_unfinished_3) {
  std::string input("real y;\n"
                    "real z;\n"
                    "int;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: <identifier>"), std::string::npos);
}

TEST(Parser, parse_unfinished_4) {
  std::string input("real y\n"
                    "real z\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED"), std::string::npos);
  EXPECT_NE(msgs.str().find("\";\""), std::string::npos);
}

TEST(Parser, parse_unfinished_matrix) {
  std::string input("matrix;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: \"[\""), std::string::npos);
}

TEST(Parser, parse_matrix_missing_commas) {
  std::string input("matrix[2 3] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: \",\""), std::string::npos);
}

TEST(Parser, parse_bounded_simplex) {
  std::string input("int K;\n"
                    "simplex<upper=5>[K] foo;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: <vector length declaration"), std::string::npos);
}

TEST(Parser, parse_unclosed_dim) {
  std::string input("int K;\n"
                    "simplex[K foo;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: \"]\""), std::string::npos);
}

TEST(Parser, parse_non_int_dim_1) {
  std::string input("real K;\n"
                    "simplex[K] foo;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("expression denoting integer required; found type=real"), std::string::npos);
}

TEST(Parser, parse_non_int_dim_2) {
  std::string input("simplex[1.1] foo;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("expression denoting integer required; found type=real"), std::string::npos);
}

TEST(Parser, parse_too_many_dims) {
  std::string input("int J;\n"
                    "int K;\n"
                    "int L;\n"
                    "matrix[J, K, L] foo;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: \"]\""), std::string::npos);
}

TEST(Parser, parse_no_dims) {
  std::string input("int J;\n"
                    "int K;\n"
                    "int L;\n"
                    "matrix[,] foo;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: <data-only integer expression"), std::string::npos);
}

TEST(Parser, redeclare) {
  std::string input("int J;\n"
                    "real J;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("duplicate declaration"), std::string::npos);
  EXPECT_NE(msgs.str().find("redeclare"), std::string::npos);

}

TEST(Parser, undeclared_size) {
  std::string input("simplex[K] a;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("variable \"K\" does not exist."), std::string::npos);
}

TEST(Parser, size_test) {
  std::string input("real K;\n"
                    "matrix[K, K] a;\n"
                    "simplex[K] b;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("expression denoting integer required"), std::string::npos);
}

TEST(Parser, bounds_test_1) {
  std::string input("real K;\n"
                    "int<lower=K> J;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("expression denoting integer required"), std::string::npos);
}

TEST(Parser, bounds_test_2) {
  std::string input("int<lower=1, upper=2.0> J;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("expression denoting integer required"), std::string::npos);
}

TEST(Parser, bounds_test_3) {
  std::string input("int<middle=1, upper=2.0> J;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED"), std::string::npos);
  EXPECT_NE(msgs.str().find("\"lower\""), std::string::npos);
  EXPECT_NE(msgs.str().find("\"upper\""), std::string::npos);
}

TEST(Parser, bounds_test_4) {
  std::string input("int<lower 1, upper=2.0> J;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED"), std::string::npos);
  EXPECT_NE(msgs.str().find("\"=\""), std::string::npos);
}

TEST(Parser, bounds_test_5) {
  std::string input("int<upper=2,lower=1> J;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED"), std::string::npos);
  EXPECT_NE(msgs.str().find("\">\""), std::string::npos);
}

TEST(Parser, bounds_test_6) {
  std::string input("int<lower=1,upper=2,upper=3> J;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED"), std::string::npos);
  EXPECT_NE(msgs.str().find("\">\""), std::string::npos);
}


//  std::cout << msgs.str() << std::endl;
