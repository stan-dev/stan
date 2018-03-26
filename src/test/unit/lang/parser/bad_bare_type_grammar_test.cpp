#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <boost/algorithm/string/predicate.hpp>

TEST(Parser, parse_local_unknown) {
  std::string input("functions {\n"
                    "unknown bad_fun; \n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fns;
  fns = parse_functions(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("PARSER FAILED TO PARSE INPUT"), std::string::npos);
}

TEST(Parser, parse_bare_unclosed_dim) {
  std::string input("functions {\n"
                    "void bad_fun(int[ ); \n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fns;
  fns = parse_functions(input, pass, msgs);

  EXPECT_FALSE(pass);
  EXPECT_NE(msgs.str().find("<comma to indicate more dimensions or ] "), std::string::npos);
}

TEST(Parser, parse_bare_unclosed_dim_2) {
  std::string input("functions {\n"
                    "void bad_fun(int[ , , , ); \n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fns;
  fns = parse_functions(input, pass, msgs);

  EXPECT_NE(msgs.str().find("<comma to indicate more dimensions or ] "), std::string::npos);
  EXPECT_FALSE(pass);

}
