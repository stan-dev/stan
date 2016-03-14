#include <stan/services/arguments/argument_parser.hpp>
#include <stan/services/arguments/arg_id.hpp>
#include <stan/services/arguments/arg_data.hpp>
#include <stan/services/arguments/arg_init.hpp>
#include <stan/services/arguments/arg_random.hpp>
#include <stan/services/arguments/arg_output.hpp>
#include <stan/interface_callbacks/writer/noop_writer.hpp>
#include <gtest/gtest.h>

using stan::services::argument;
using stan::services::arg_id;
using stan::services::arg_data;
using stan::services::arg_init;
using stan::services::arg_random;
using stan::services::arg_output;
using stan::services::argument_parser;
using stan::services::error_codes;

class StanServicesArgumentsArgumentParser : public testing::Test {
public:
  void SetUp() {
    parser = new argument_parser();
    parser->push_valid_arg(new arg_id());
    parser->push_valid_arg(new arg_data());
    parser->push_valid_arg(new arg_init());
    parser->push_valid_arg(new arg_random());
    parser->push_valid_arg(new arg_output());
  }
  void TearDown() {
    delete(parser);
  }
  
  std::vector<argument*> valid_arguments;
  argument_parser* parser;
  int err_code;
  stan::interface_callbacks::writer::noop_writer writer;
};

 
TEST_F(StanServicesArgumentsArgumentParser, default) {
  const char* argv[] = {};
  int argc = 0;
  
  err_code = parser->parse_args(argc, argv, writer, writer);
  EXPECT_EQ(int(error_codes::USAGE), err_code);
}

TEST_F(StanServicesArgumentsArgumentParser, help) {
  const char* argv[] = {"model_name", "help"};
  int argc = 2;
  
  err_code = parser->parse_args(argc, argv, writer, writer);
  EXPECT_EQ(int(error_codes::OK), err_code);
}

TEST_F(StanServicesArgumentsArgumentParser, unrecognized_argument) {
  const char* argv[] = {"foo"};
  int argc = 1;
  
  err_code = parser->parse_args(argc, argv, writer, writer);
  EXPECT_EQ(int(error_codes::USAGE), err_code);
}
