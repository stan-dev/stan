#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <gtest/gtest.h>

// parse program, then generate log_prop function

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

  stan::lang::generate_log_prob(prgrm, cpp_code)
;
  std::cout << "test.hpp:" << std::endl;
  std::cout << cpp_code.str() << std::endl;

  return pass;
}

TEST(generateCtor, ex_model_kidiq_model) {
  std::string input("data {\n"
                    "  int<lower=0> N;\n"
                    "  vector<lower=0, upper=200>[N] kid_score;\n"
                    "  vector<lower=0, upper=200>[N] mom_iq;\n"
                    "  vector<lower=0, upper=1>[N] mom_hs;\n"
                    "}\n"
                    "parameters {\n"
                    "  real p_y;\n"
                    "  real p_d4_y[2,3,4,5];\n"
                    "  matrix[2,3] p_d2_mat[4,5];\n"
                    "  corr_matrix[2] p_d2_corr_mat[4,5];\n"
                    "  vector[3] beta;\n"
                    "  real<lower=0> sigma;\n"
                    "}\n"
                    "model {\n"
                    "  sigma ~ cauchy(0, 2.5);\n"
                    "  kid_score ~ normal(beta[1] + beta[2] * mom_hs + beta[3] * mom_iq, sigma);\n"
                    "}\n");

  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS
}


