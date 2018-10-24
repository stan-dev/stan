#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

TEST(lang, deep_copy_hpp) {
  std::string m1("functions{\n"
                 "  matrix covsqrt2corsqrt(matrix mat, int invert){\n"
                 "    matrix[rows(mat),cols(mat)] o;\n"
                 "    o=mat;\n"
                 "    o[1] = o[2];\n"
                 "    o[3:4] = o[1:2];\n"
                 "    return o;\n"
                 "  }\n"
                 "}\n\n");
  expect_matches(1,m1,"stan::model::deep_copy");
}
