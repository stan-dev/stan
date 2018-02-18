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
