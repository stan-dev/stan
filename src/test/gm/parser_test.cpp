#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <istream>


#include "stan/gm/ast.hpp"
#include "stan/gm/parser.hpp"

bool is_parsable(const std::string& file_name) {
  stan::gm::program prog;
  std::ifstream fs(file_name.c_str());
  return stan::gm::parse(fs, file_name, prog);
}

TEST(GmParser, NormalExample) {
  EXPECT_TRUE(is_parsable("src/models/eight_schools.stan"));
}
