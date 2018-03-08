#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <gtest/gtest.h>

// parse program, then generate write array method

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

  stan::lang::generate_write_array_method(prgrm, "unit_test_model", cpp_code)
    ;
  std::cout << "test.hpp:" << std::endl;
  std::cout << cpp_code.str() << std::endl;

  return pass;
}

TEST(generateWriteArray, ex_model_kidiq_decls) {
  std::string input("data {\n"
                    "  int<lower=0> N;\n"
                    "  vector<lower=0, upper=200>[N] kid_score;\n"
                    "  real<lower=0, upper=1> mom_hs_new;\n"
                    "  real<lower=0, upper=200> mom_iq_new;\n"
                    "}\n"
                    "parameters {\n"
                    "  vector[3] beta;\n"
                    "  real<lower=0> sigma;\n"
                    "}\n"
                    "generated quantities {\n"
                    "  real kid_score_pred;\n"
                    "  kid_score_pred = normal_rng(beta[1] + beta[2] * mom_hs_new\n"
                    "                               + beta[3] * mom_iq_new, sigma);\n"
                    "}\n");
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS
}


TEST(generateWriteArray, test2) {
  std::string input(
                    "data {\n"
                    "  int<lower=0> N;\n"
                    "}\n"
                    "parameters {\n"
                    "  real y;\n"
                    "  real<lower=0> p_r1;\n"
                    "  vector[N] p_v1;\n"
                    "  vector[10] p_ar_vec[5];\n"
                    "  vector[10] p_d2_ar_vec[5,6];\n"
                    "  vector[10] p_d3_ar_vec[5,6,7];\n"
                    "  matrix<lower=-7,upper=6>[3, 6] my_mat_mn;\n"
                    "  matrix<upper=0>[2,4] d2_array_of_mat[8, 9];\n"
                    "\n"
                    "  simplex[5] s5;\n"
                    "  corr_matrix[5] cm5;\n"
                    "}\n"
                    "transformed parameters {\n"
                    "  real tp_ar_r[4];\n"
                    "  vector[10] tp_ar_vec[5];\n"
                    "  vector[10] tp_d2_ar_vec[5,6];\n"
                    "  vector[10] tp_d3_ar_vec[5,6,7];\n"
                    "\n"
                    "  real tp_y;\n"
                    "  real<lower=0> tp_r1;\n"
                    "  vector[N] tp_v1;\n"
                    "}\n"
                    "model {\n"
                    "  y ~ normal(0,1);\n"
                    "}\n"
                    "generated quantities {\n"
                    "  int gq_i;\n"
                    "  int gq_ar_i[3];\n"
                    "  real gq_ar_r[4];\n"
                    "  vector[10] gq_ar_vec[5];\n"
                    "  vector[10] gq_d2_ar_vec[5,6];\n"
                    "  vector[10] gq_d3_ar_vec[5,6,7];\n"
                    "}\n"
                    "\n");

  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS
}
  



TEST(generateWriteArray, test3) {
  std::string input(
                    "data {\n"
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
                    "  real p_y;\n"
                    "  real p_d4_y[2,3,4,5];\n"
                    "  matrix[2,3] p_d2_mat[4,5];\n"
                    "  corr_matrix[2] p_d2_corr_mat[4,5];\n"
                    "  real<lower=0> sigma;\n"
                    "  simplex[5] p_simplex;\n"
                    "}\n"
                    "transformed parameters {\n"
                    "  real tp_y;\n"
                    "  real tp_d4_y[2,3,4,5];\n"
                    "  matrix[2,3] tp_d2_mat[4,5];\n"
                    "  corr_matrix[2] tp_d2_corr_mat[4,5];\n"
                    "  simplex[6] tp_simplex;\n"
                    "  {\n"
                    "    int i = 200;\n"
                    "    print(i);\n"
                    "  }\n"
                    "}\n"
                    "model {\n"
                    "  p_y ~ normal(0, 1);\n"
                    "}\n"
                    "generated quantities {\n"
                    "    int i = 200;\n"
                    "    print(i);\n"
                    "}  \n"
                    "\n");
  
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS
}
