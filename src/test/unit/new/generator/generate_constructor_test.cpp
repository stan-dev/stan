#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <gtest/gtest.h>

// parse program, then generate constructor

bool run_test(std::string& stan_code,
              std::stringstream& cpp_code) {

  bool pass = false;
  std::stringstream msgs;
  stan::lang::program prgrm;
  prgrm = parse_program(stan_code, pass, msgs);
  if (!pass) {
    cpp_code << msgs.str();
    return pass;
  }
  std::cout << "\ntest.stan:" << std::endl;
  std::cout << stan_code << std::endl;

  stan::lang::generate_constructor(prgrm, "unit_test_model", cpp_code)
;
  std::cout << "test.hpp:" << std::endl;
  std::cout << cpp_code.str() << std::endl;

  return pass;
}

TEST(generateCtor, ex_model_kidiq_decls) {
  std::string input("data {\n"
                    "  int<lower=0> N;\n"
                    "  vector<lower=0, upper=200>[N] kid_score;\n"
                    "  real<lower=0, upper=1> mom_hs_new;\n"
                    "}\n"
                    "parameters {\n"
                    "  vector[3] beta;\n"
                    "  real<lower=0> sigma;\n"
                    "}\n");
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS
}

TEST(generateCtor, d_td_p_tp_decls) {
  std::string input("data {\n"
                    "  int<lower=0> N;\n"
                    "  real d_x;\n"
                    "  matrix[2,3] d_d2_mat[4,5];\n"
                    "  corr_matrix[2] d_d2_corr_mat[4,5];\n"
                    "}\n"
                    "transformed data {\n"
                    "  real td_x;\n"
                    "  matrix[6,7] td_d2_mat[8,9];\n"
                    "  corr_matrix[2] td_d2_corr_mat[11,12];\n"
                    "}\n"
                    "parameters {\n"
                    "  real p_x;\n"
                    "  matrix[2,3] p_d2_mat[4,5];\n"
                    "  corr_matrix[2] p_d2_corr_mat[4,5];\n"
                    "  real<lower=0> sigma;\n"
                    "}\n");
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS
}


TEST(generateCtor, td_defs) {
  std::string input("transformed data {\n"
                    "  real td_x = 5.0;\n"
                    "  matrix[6,7] td_d2_mat[8,9];\n"
                    "  corr_matrix[2] td_d2_corr_mat[11,12];\n"
                    "}\n");
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS
}


