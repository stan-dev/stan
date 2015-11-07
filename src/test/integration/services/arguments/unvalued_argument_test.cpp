#include <stan/services/arguments/unvalued_argument.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/interface_callbacks/writer/noop_writer.hpp>
#include <gtest/gtest.h>

class test_arg_impl : public stan::services::unvalued_argument {
  std::string print_value() {
    return "";
  }
  std::string print_valid() {
    return "";
  }
  bool is_default() {
    return true;
  }
};


class StanServicesArgumentsUnvaluedArgument : public testing::Test {
public:
  void SetUp () {
    arg = new test_arg_impl;
  }
  void TearDown() {
    delete(arg);
  }
  
  stan::services::argument *arg;
  std::stringstream ss;
};


TEST_F(StanServicesArgumentsUnvaluedArgument,Constructor) {
  // test fixture would have created the argument.
}

TEST_F(StanServicesArgumentsUnvaluedArgument,name) {
  EXPECT_EQ("", arg->name());
}

TEST_F(StanServicesArgumentsUnvaluedArgument,description) {
  EXPECT_EQ("", arg->description());
}

TEST_F(StanServicesArgumentsUnvaluedArgument,print) {
  // FIXME: write test
}

TEST_F(StanServicesArgumentsUnvaluedArgument,print_help) {
  // FIXME: write test
}

TEST_F(StanServicesArgumentsUnvaluedArgument,parse_args) {
  bool return_value;
  std::vector<std::string> args;
  bool help_flag;
  stan::interface_callbacks::writer::stream_writer out(ss);
  stan::interface_callbacks::writer::noop_writer err;
  
  return_value = false;
  args.clear();
  help_flag = false;
  return_value = arg->parse_args(args,out,err,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(0U, args.size());
  
  
  return_value = false;
  args.clear();
  args.push_back("help");
  help_flag = false;
  return_value = arg->parse_args(args,out,err,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_TRUE(help_flag);
  EXPECT_EQ(0U, args.size());
  EXPECT_FALSE(static_cast<test_arg_impl*>(arg)->is_present());
  
  
  return_value = false;
  args.clear();
  args.push_back("help-all");
  help_flag = false;
  return_value = arg->parse_args(args,out,err,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_TRUE(help_flag);
  EXPECT_EQ(0U, args.size());
  EXPECT_FALSE(static_cast<test_arg_impl*>(arg)->is_present());
}

TEST_F(StanServicesArgumentsUnvaluedArgument,parse_args_unexpected) {
  bool return_value;
  std::vector<std::string> args;
  bool help_flag;
  stan::interface_callbacks::writer::stream_writer out(ss);
  stan::interface_callbacks::writer::noop_writer err;

  return_value = false;
  args.clear();
  args.push_back("foo=bar");
  help_flag = false;
  return_value = arg->parse_args(args,out,err,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(1U, args.size());
  EXPECT_TRUE(static_cast<test_arg_impl*>(arg)->is_present());
}

TEST_F(StanServicesArgumentsUnvaluedArgument,arg) {
  EXPECT_EQ(0, arg->arg(""));
  EXPECT_EQ(0, arg->arg("foo"));
}
