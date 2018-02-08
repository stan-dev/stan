#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <boost/algorithm/string/predicate.hpp>

TEST(Parser, parse_local_unknown) {
  std::string input("unknown x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER FAILED TO PARSE INPUT"), std::string::npos);
}

TEST(Parser, parse_local_unfinished_1) {
  std::string input("int;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);
  
  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: <identifier>"), std::string::npos);
}

TEST(Parser, parse_local_unfinished_2) {
  std::string input("int x");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);
  
  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED"), std::string::npos);
  EXPECT_NE(msgs.str().find("\";\""), std::string::npos);
}

TEST(Parser, parse_local_unfinished_3) {
  std::string input("real y;\n"
                    "real z;\n"
                    "int;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: <identifier>"), std::string::npos);
}

TEST(Parser, parse_local_unfinished_4) {
  std::string input("real y\n"
                    "real z\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED"), std::string::npos);
  EXPECT_NE(msgs.str().find("\";\""), std::string::npos);
}

TEST(Parser, parse_local_unfinished_matrix) {
  std::string input("matrix;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: \"[\""), std::string::npos);
}

TEST(Parser, parse_local_matrix_missing_commas) {
  std::string input("matrix[2 3] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: \",\""), std::string::npos);
}

TEST(Parser, parse_local_unclosed_dim) {
  std::string input("int K;\n"
                    "vector[K foo;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: \"]\""), std::string::npos);
}

TEST(Parser, parse_local_non_int_dim_1) {
  std::string input("real K;\n"
                    "vector[K] foo;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("expression denoting integer required; found type=real"), std::string::npos);
}

TEST(Parser, parse_local_non_int_dim_2) {
  std::string input("vector[1.1] foo;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("expression denoting integer required; found type=real"), std::string::npos);
}

TEST(Parser, parse_local_too_many_dims) {
  std::string input("int J;\n"
                    "int K;\n"
                    "int L;\n"
                    "matrix[J, K, L] foo;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: \"]\""), std::string::npos);
}

TEST(Parser, parse_local_no_dims) {
  std::string input("int J;\n"
                    "int K;\n"
                    "int L;\n"
                    "matrix[,] foo;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER EXPECTED: <data-only integer expression"), std::string::npos);
}

TEST(Parser, redeclare) {
  std::string input("int J;\n"
                    "real J;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("duplicate declaration"), std::string::npos);
  EXPECT_NE(msgs.str().find("redeclare"), std::string::npos);

}

TEST(Parser, undeclared_size) {
  std::string input("vector[K] a;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("variable \"K\" does not exist."), std::string::npos);
}

TEST(Parser, size_test) {
  std::string input("real K;\n"
                    "matrix[K, K] a;\n"
                    "row_vector[K] b;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("expression denoting integer required"), std::string::npos);
}
