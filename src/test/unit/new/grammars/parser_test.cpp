#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

TEST(Parser, program1) {
  std::string input("parameters {\n"
                    "real y;\n"
                    "}\n"
                    "model {\n"
                    "  y ~ normal(0,1);\n"
                    "}\n");

  bool pass = false;
  std::stringstream msgs;
  stan::lang::program prgrm;
  prgrm = parse_program(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}

TEST(Parser, program2) {
  std::string input("data {\n"
                    "  int N;\n"
                    "  vector[N] x;\n"
                    "  vector[N] y;\n"
                    "}\n"
                    "parameters {\n"
                    "  real alpha;\n"
                    "  real beta;\n"
                    "  real sigma;\n"
                    "}\n"
                    "model {\n"
                    "  y ~ normal(alpha + x * beta, sigma);\n"
                    "  alpha ~ normal(0, 5);\n"
                    "  beta ~ normal(0, 5);\n"
                    "  sigma ~ cauchy(0, 2.5);\n"
                    "}\n");

  bool pass = false;
  std::stringstream msgs;
  stan::lang::program prgrm;
  prgrm = parse_program(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}

TEST(Parser, ex_mode_kidiq_predict) {
  std::string input("data {\n"
                    "  int<lower=0> N;\n"
                    "  vector<lower=0, upper=200>[N] kid_score;\n"
                    "  vector<lower=0, upper=200>[N] mom_iq;\n"
                    "  vector<lower=0, upper=1>[N] mom_hs;\n"
                    "  real<lower=0, upper=1> mom_hs_new;           // for prediction\n"
                    "  real<lower=0, upper=200> mom_iq_new;\n"
                    "}\n"
                    "parameters {\n"
                    "  vector[3] beta;\n"
                    "  real<lower=0> sigma;\n"
                    "}\n"
                    "model {\n"
                    "  sigma ~ cauchy(0, 2.5);\n"
                    "  kid_score ~ normal(beta[1] + beta[2] * mom_hs + beta[3] * mom_iq, sigma);\n"
                    "}\n"
                    "generated quantities {       // prediction\n"
                    "  real kid_score_pred;\n"
                    "  kid_score_pred = normal_rng(beta[1] + beta[2] * mom_hs_new\n"
                    "                               + beta[3] * mom_iq_new, sigma);\n"
                    "}\n");

  bool pass = false;
  std::stringstream msgs;
  stan::lang::program prgrm;
  prgrm = parse_program(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}


TEST(Parser, ex_mode_normal_multi) {
  std::string input("data {\n"
                    "  int<lower=0> N; //response variable size\n"
                    "  int<lower=0> J; //number of observations\n"
                    "  vector[N] y[J];\n"
                    "  vector[N] z[J];\n"
                    "  matrix[N, N] sigma;\n"
                    "}\n"
                    "transformed data {\n"
                    "  row_vector[N] ry[J];\n"
                    "  row_vector[N] rz[J];\n"
                    "  matrix[N, N] inv_sigma;\n"
                    "  for (n in 1:N)\n"
                    "    for (j in 1:J) {\n"
                    "      ry[j][n] = y[j][n];\n"
                    "      rz[j][n] = z[j][n];\n"
                    "    }\n"
                    "  inv_sigma = inverse_spd(sigma);\n"
                    "}\n"
                    "parameters {\n"
                    "  vector[N] beta[J];\n"
                    "  row_vector[N] rbeta[J];\n"
                    "}\n"
                    "model {\n"
                    "  y ~ multi_normal(beta, sigma);\n"
                    "  y ~ multi_normal(beta[1], sigma);\n"
                    "  y[1] ~ multi_normal(beta, sigma);\n"
                    "  y[1] ~ multi_normal(beta[1], sigma);\n"
                    "  y ~ multi_normal(z, sigma);\n"
                    "  y ~ multi_normal(z[1], sigma);\n"
                    "  y[1] ~ multi_normal(z, sigma);\n"
                    "  y[1] ~ multi_normal(z[1], sigma);\n"
                    "  \n"
                    "  y ~ multi_normal(rbeta, sigma);\n"
                    "  y ~ multi_normal(rbeta[1], sigma);\n"
                    "  y[1] ~ multi_normal(rbeta, sigma);\n"
                    "  y[1] ~ multi_normal(rbeta[1], sigma);\n"
                    "  y ~ multi_normal(rz, sigma);\n"
                    "  y ~ multi_normal(rz[1], sigma);\n"
                    "  y[1] ~ multi_normal(rz, sigma);\n"
                    "  y[1] ~ multi_normal(rz[1], sigma);\n"
                    "  \n"
                    "  ry ~ multi_normal(beta, sigma);\n"
                    "  ry ~ multi_normal(beta[1], sigma);\n"
                    "  ry[1] ~ multi_normal(beta, sigma);\n"
                    "  ry[1] ~ multi_normal(beta[1], sigma);\n"
                    "  ry ~ multi_normal(z, sigma);\n"
                    "  ry ~ multi_normal(z[1], sigma);\n"
                    "  ry[1] ~ multi_normal(z, sigma);\n"
                    "  ry[1] ~ multi_normal(z[1], sigma);\n"
                    "  \n"
                    "  ry ~ multi_normal(rbeta, sigma);\n"
                    "  ry ~ multi_normal(rbeta[1], sigma);\n"
                    "  ry[1] ~ multi_normal(rbeta, sigma);\n"
                    "  ry[1] ~ multi_normal(rbeta[1], sigma);\n"
                    "  ry ~ multi_normal(rz, sigma);\n"
                    "  ry ~ multi_normal(rz[1], sigma);\n"
                    "  ry[1] ~ multi_normal(rz, sigma);\n"
                    "  ry[1] ~ multi_normal(rz[1], sigma);\n"
                    "\n"
                    "\n"
                    "\n"
                    "  y ~ multi_normal_cholesky(beta, sigma);\n"
                    "  y ~ multi_normal_cholesky(beta[1], sigma);\n"
                    "  y[1] ~ multi_normal_cholesky(beta, sigma);\n"
                    "  y[1] ~ multi_normal_cholesky(beta[1], sigma);\n"
                    "  y ~ multi_normal_cholesky(z, sigma);\n"
                    "  y ~ multi_normal_cholesky(z[1], sigma);\n"
                    "  y[1] ~ multi_normal_cholesky(z, sigma);\n"
                    "  y[1] ~ multi_normal_cholesky(z[1], sigma);\n"
                    "  \n"
                    "  y ~ multi_normal_cholesky(rbeta, sigma);\n"
                    "  y ~ multi_normal_cholesky(rbeta[1], sigma);\n"
                    "  y[1] ~ multi_normal_cholesky(rbeta, sigma);\n"
                    "  y[1] ~ multi_normal_cholesky(rbeta[1], sigma);\n"
                    "  y ~ multi_normal_cholesky(rz, sigma);\n"
                    "  y ~ multi_normal_cholesky(rz[1], sigma);\n"
                    "  y[1] ~ multi_normal_cholesky(rz, sigma);\n"
                    "  y[1] ~ multi_normal_cholesky(rz[1], sigma);\n"
                    "  \n"
                    "  ry ~ multi_normal_cholesky(beta, sigma);\n"
                    "  ry ~ multi_normal_cholesky(beta[1], sigma);\n"
                    "  ry[1] ~ multi_normal_cholesky(beta, sigma);\n"
                    "  ry[1] ~ multi_normal_cholesky(beta[1], sigma);\n"
                    "  ry ~ multi_normal_cholesky(z, sigma);\n"
                    "  ry ~ multi_normal_cholesky(z[1], sigma);\n"
                    "  ry[1] ~ multi_normal_cholesky(z, sigma);\n"
                    "  ry[1] ~ multi_normal_cholesky(z[1], sigma);\n"
                    "  \n"
                    "  ry ~ multi_normal_cholesky(rbeta, sigma);\n"
                    "  ry ~ multi_normal_cholesky(rbeta[1], sigma);\n"
                    "  ry[1] ~ multi_normal_cholesky(rbeta, sigma);\n"
                    "  ry[1] ~ multi_normal_cholesky(rbeta[1], sigma);\n"
                    "  ry ~ multi_normal_cholesky(rz, sigma);\n"
                    "  ry ~ multi_normal_cholesky(rz[1], sigma);\n"
                    "  ry[1] ~ multi_normal_cholesky(rz, sigma);\n"
                    "  ry[1] ~ multi_normal_cholesky(rz[1], sigma);\n"
                    "\n"
                    "  y ~ multi_normal_prec(beta, sigma);\n"
                    "  y ~ multi_normal_prec(beta[1], sigma);\n"
                    "  y[1] ~ multi_normal_prec(beta, sigma);\n"
                    "  y[1] ~ multi_normal_prec(beta[1], sigma);\n"
                    "  y ~ multi_normal_prec(z, sigma);\n"
                    "  y ~ multi_normal_prec(z[1], sigma);\n"
                    "  y[1] ~ multi_normal_prec(z, sigma);\n"
                    "  y[1] ~ multi_normal_prec(z[1], sigma);\n"
                    "\n"
                    "  y ~ multi_normal_prec(rbeta, sigma);\n"
                    "  y ~ multi_normal_prec(rbeta[1], sigma);\n"
                    "  y[1] ~ multi_normal_prec(rbeta, sigma);\n"
                    "  y[1] ~ multi_normal_prec(rbeta[1], sigma);\n"
                    "  y ~ multi_normal_prec(rz, sigma);\n"
                    "  y ~ multi_normal_prec(rz[1], sigma);\n"
                    "  y[1] ~ multi_normal_prec(rz, sigma);\n"
                    "  y[1] ~ multi_normal_prec(rz[1], sigma);\n"
                    "  \n"
                    "  ry ~ multi_normal_prec(beta, sigma);\n"
                    "  ry ~ multi_normal_prec(beta[1], sigma);\n"
                    "  ry[1] ~ multi_normal_prec(beta, sigma);\n"
                    "  ry[1] ~ multi_normal_prec(beta[1], sigma);\n"
                    "  ry ~ multi_normal_prec(z, sigma);\n"
                    "  ry ~ multi_normal_prec(z[1], sigma);\n"
                    "  ry[1] ~ multi_normal_prec(z, sigma);\n"
                    "  ry[1] ~ multi_normal_prec(z[1], sigma);\n"
                    "  \n"
                    "  ry ~ multi_normal_prec(rbeta, sigma);\n"
                    "  ry ~ multi_normal_prec(rbeta[1], sigma);\n"
                    "  ry[1] ~ multi_normal_prec(rbeta, sigma);\n"
                    "  ry[1] ~ multi_normal_prec(rbeta[1], sigma);\n"
                    "  ry ~ multi_normal_prec(rz, sigma);\n"
                    "  ry ~ multi_normal_prec(rz[1], sigma);\n"
                    "  ry[1] ~ multi_normal_prec(rz, sigma);\n"
                    "  ry[1] ~ multi_normal_prec(rz[1], sigma);\n"
                    "  \n"
                    "  y ~ multi_student_t(10, beta, sigma);\n"
                    "  y ~ multi_student_t(10, beta[1], sigma);\n"
                    "  y[1] ~ multi_student_t(10, beta, sigma);\n"
                    "  y[1] ~ multi_student_t(10, beta[1], sigma);\n"
                    "  y ~ multi_student_t(10, z, sigma);\n"
                    "  y ~ multi_student_t(10, z[1], sigma);\n"
                    "  y[1] ~ multi_student_t(10, z, sigma);\n"
                    "  y[1] ~ multi_student_t(10, z[1], sigma);\n"
                    "  \n"
                    "  y ~ multi_student_t(10, rbeta, sigma);\n"
                    "  y ~ multi_student_t(10, rbeta[1], sigma);\n"
                    "  y[1] ~ multi_student_t(10, rbeta, sigma);\n"
                    "  y[1] ~ multi_student_t(10, rbeta[1], sigma);\n"
                    "  y ~ multi_student_t(10, rz, sigma);\n"
                    "  y ~ multi_student_t(10, rz[1], sigma);\n"
                    "  y[1] ~ multi_student_t(10, rz, sigma);\n"
                    "  y[1] ~ multi_student_t(10, rz[1], sigma);\n"
                    "  \n"
                    "  ry ~ multi_student_t(10, beta, sigma);\n"
                    "  ry ~ multi_student_t(10, beta[1], sigma);\n"
                    "  ry[1] ~ multi_student_t(10, beta, sigma);\n"
                    "  ry[1] ~ multi_student_t(10, beta[1], sigma);\n"
                    "  ry ~ multi_student_t(10, z, sigma);\n"
                    "  ry ~ multi_student_t(10, z[1], sigma);\n"
                    "  ry[1] ~ multi_student_t(10, z, sigma);\n"
                    "  ry[1] ~ multi_student_t(10, z[1], sigma);\n"
                    "  \n"
                    "  ry ~ multi_student_t(10, rbeta, sigma);\n"
                    "  ry ~ multi_student_t(10, rbeta[1], sigma);\n"
                    "  ry[1] ~ multi_student_t(10, rbeta, sigma);\n"
                    "  ry[1] ~ multi_student_t(10, rbeta[1], sigma);\n"
                    "  ry ~ multi_student_t(10, rz, sigma);\n"
                    "  ry ~ multi_student_t(10, rz[1], sigma);\n"
                    "  ry[1] ~ multi_student_t(10, rz, sigma);\n"
                    "  ry[1] ~ multi_student_t(10, rz[1], sigma);\n"
                    "}\n");

  bool pass = false;
  std::stringstream msgs;
  stan::lang::program prgrm;
  prgrm = parse_program(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string(""));
}
