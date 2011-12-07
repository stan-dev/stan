#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <exception>

#include "stan/gm/ast.hpp"
#include "stan/gm/parser.hpp"

bool is_parsable(const std::string& file_name) {
  stan::gm::program prog;
  std::ifstream fs(file_name.c_str());
  return stan::gm::parse(fs, file_name, prog);
}

TEST(gm_parser,parsable_demos) {
  EXPECT_TRUE(is_parsable("src/models/eight_schools.stan"));
}

TEST(gm_parser,parsable_test_bad1) {
  EXPECT_THROW(is_parsable("src/test/gm/model_specs/bad1.stan"),
	       std::runtime_error);
}
TEST(gm_parser,parsable_test_bad2) {
  EXPECT_THROW(is_parsable("src/test/gm/model_specs/bad2.stan"),
	       std::runtime_error);
}

TEST(gm_parser,parsable_test_bad3) {
  EXPECT_THROW(is_parsable("src/test/gm/model_specs/bad3.stan"),
	       std::runtime_error);
}

TEST(gm_parser,parsable_test_bad4) {
  EXPECT_THROW(is_parsable("src/test/gm/model_specs/bad3.stan"),
	       std::runtime_error);
}
