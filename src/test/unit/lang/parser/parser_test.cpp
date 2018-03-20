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
