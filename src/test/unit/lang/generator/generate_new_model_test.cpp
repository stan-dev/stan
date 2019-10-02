#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <test/unit/lang/utility.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>

TEST(langGenerator, generateNewModel) {
  stan::lang::program prog;
  std::string model_name = "m";
  std::stringstream code_stream;

  stan::io::program_reader reader;
  // fake reader history - no stan program, just AST
  reader.add_event(0, 0, "start", "generator-test");
  reader.add_event(500, 500, "end", "generator-test");

  stan::lang::generate_cpp(prog, model_name, reader.history(), code_stream);
  std::string generated_code = code_stream.str();
  EXPECT_EQ(
      1,
      count_matches(
          "#ifndef USING_R\n\n"
          "stan::model::model_base& new_model(\n"
          "        stan::io::var_context& data_context,\n"
          "        unsigned int seed,\n"
          "        std::ostream* msg_stream) {\n"
          "  stan_model* m = new stan_model(data_context, seed, msg_stream);\n"
          "  return *m;\n"
          "}\n\n"
          "#endif\n",
          generated_code));
}
