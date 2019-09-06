#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/utility.hpp>

#include <stan/callbacks/stream_writer.hpp>
#include <stan/io/dump.hpp>

#include <test/test-models/good/parser-generator/deep-copy-warning.hpp>

#include <boost/random/additive_combine.hpp>
#include <fstream>
#include <stdexcept>
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

TEST(lang, run_deep_copy) {
  std::fstream empty_data_stream("");
  stan::io::dump empty_data_context(empty_data_stream);
  empty_data_stream.close();
  std::stringstream model_output;

  // instantiate good test model parser-generator/deep-copy-warning
  stan_model model(empty_data_context, &model_output);

  // check print stmts
  EXPECT_EQ(1,count_matches("y: 0.1",model_output.str()));
  EXPECT_EQ(1,count_matches("v: [2,2,3,4,5]",model_output.str()));
  EXPECT_EQ(1,count_matches("v: [3,3,3,4,5]",model_output.str()));
  EXPECT_EQ(1,count_matches("v: [3,7,7,8,5]",model_output.str()));
}
