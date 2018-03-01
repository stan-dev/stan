#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <gtest/gtest.h>

// use block_var_decls grammar utility to generate list of bvds
// then check generated code

bool run_test(std::string& stan_code,
              std::stringstream& cpp_code) {

  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(stan_code, pass, msgs);
  if (!pass) {
    cpp_code << msgs.str();
    return pass;
  }
  std::cout << "\ntest.stan:" << std::endl;
  std::cout << stan_code << std::endl;

  stan::lang::generate_validate_var_decls(bvds, 1, cpp_code);

  std::cout << "test.hpp:" << std::endl;
  std::cout << cpp_code.str() << std::endl;

  return pass;
}

TEST(generateValidateVarDecls, t1) {
  std::string input("int N;\n"
                    "real y;\n"
                    "real<lower=2.1, upper=2.9> z;\n");
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS:  1 bounded decl
}

TEST(generateValidateVarDecls, t2) {
  std::string input("int<lower=0> N;\n"
                    "int<lower=0> M;\n"
                    "int<lower=0> I;\n"
                    "int<lower=0> J;\n"
                    "matrix<lower=-7,upper=6>[M, N] my_mat_mn;\n"
                    "matrix<lower=M,upper=I>[M,N] d1_array_of_mat[I];\n"
                    "matrix<upper=J>[M,N] d2_array_of_mat[I, J];\n");

  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS?
}

TEST(generateValidateVarDecls, t3) {
  std::string input("vector<upper=0.001>[5] c[10,20,30];");

  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS?
}

TEST(generateValidateVarDecls, t4) {
  std::string input("matrix<lower=0.0, upper=1.0>[1,2] x;\n"
                    "matrix<lower=0.0, upper=1.0>[1,2] y[5,6];\n");

  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS?
}

TEST(generateValidateVarDecls, t5) {
  std::string input("simplex[5] s5;\n"
                    "corr_matrix[5] cm5;\n");

  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS?
}
