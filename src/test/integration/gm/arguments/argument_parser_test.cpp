#include <stan/gm/arguments/argument_parser.hpp>
#include <stan/gm/arguments/arg_id.hpp>
#include <stan/gm/arguments/arg_data.hpp>
#include <stan/gm/arguments/arg_init.hpp>
#include <stan/gm/arguments/arg_random.hpp>
#include <stan/gm/arguments/arg_output.hpp>
#include <gtest/gtest.h>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/null.hpp>

using stan::gm::argument;
using stan::gm::arg_id;
using stan::gm::arg_data;
using stan::gm::arg_init;
using stan::gm::arg_random;
using stan::gm::arg_output;
using stan::gm::argument_parser;
using stan::gm::error_codes;

class StanGmArgumentsArgumentParser : public testing::Test {
public:
  void SetUp() {
    // copied setup from src/stan/common/command.hpp
    // FIXME: move to factory?
    valid_arguments.push_back(new arg_id());
    valid_arguments.push_back(new arg_data());
    valid_arguments.push_back(new arg_init());
    valid_arguments.push_back(new arg_random());
    valid_arguments.push_back(new arg_output());
    
    parser = new argument_parser(valid_arguments);
  }
  void TearDown() {
    for (size_t i = 0; i < valid_arguments.size(); ++i)
      delete valid_arguments.at(i);
    delete(parser);
  }
  
  std::vector<argument*> valid_arguments;
  argument_parser* parser;
  int err_code;
  boost::iostreams::stream< boost::iostreams::null_sink > null_ostream;
};

 
TEST_F(StanGmArgumentsArgumentParser, default) {
  const char* argv[] = {};
  int argc = 0;
  
  err_code = parser->parse_args(argc, argv, &null_ostream, &null_ostream);
  EXPECT_EQ(int(error_codes::USAGE), err_code);
}

TEST_F(StanGmArgumentsArgumentParser, help) {
  const char* argv[] = {"model_name", "help"};
  int argc = 2;
  
  err_code = parser->parse_args(argc, argv, &null_ostream, &null_ostream);
  EXPECT_EQ(int(error_codes::OK), err_code);
}

TEST_F(StanGmArgumentsArgumentParser, unrecognized_argument) {
  const char* argv[] = {"foo"};
  int argc = 1;
  
  err_code = parser->parse_args(argc, argv, &null_ostream, &null_ostream);
  EXPECT_EQ(int(error_codes::USAGE), err_code);
}
