#include <stan/gm/arguments/unvalued_argument.hpp>
#include <gtest/gtest.h>

class test_arg_impl : public stan::gm::unvalued_argument {
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


class StanGmArgumentsUnvaluedArgument : public testing::Test {
public:
  void SetUp () {
    arg = new test_arg_impl;
  }
  void TearDown() {
    delete(arg);
  }
  
  stan::gm::argument *arg;
};


TEST_F(StanGmArgumentsUnvaluedArgument,Constructor) {
  // test fixture would have created the argument.
}

TEST_F(StanGmArgumentsUnvaluedArgument,name) {
  EXPECT_EQ("", arg->name());
}

TEST_F(StanGmArgumentsUnvaluedArgument,description) {
  EXPECT_EQ("", arg->description());
}

TEST_F(StanGmArgumentsUnvaluedArgument,print) {
  // FIXME: write test
}

TEST_F(StanGmArgumentsUnvaluedArgument,print_help) {
  // FIXME: write test
}

TEST_F(StanGmArgumentsUnvaluedArgument,parse_args) {
  bool return_value;
  std::vector<std::string> args;
  bool help_flag;
  
  return_value = false;
  args.clear();
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(0U, args.size());
  
  
  return_value = false;
  args.clear();
  args.push_back("help");
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_TRUE(help_flag);
  EXPECT_EQ(0U, args.size());
  EXPECT_FALSE(static_cast<test_arg_impl*>(arg)->is_present());
  
  
  return_value = false;
  args.clear();
  args.push_back("help-all");
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_TRUE(help_flag);
  EXPECT_EQ(0U, args.size());
  EXPECT_FALSE(static_cast<test_arg_impl*>(arg)->is_present());
}

TEST_F(StanGmArgumentsUnvaluedArgument,parse_args_unexpected) {
  bool return_value;
  std::vector<std::string> args;
  bool help_flag;

  return_value = false;
  args.clear();
  args.push_back("foo=bar");
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(1U, args.size());
  EXPECT_TRUE(static_cast<test_arg_impl*>(arg)->is_present());
}

TEST_F(StanGmArgumentsUnvaluedArgument,arg) {
  EXPECT_EQ(0, arg->arg(""));
  EXPECT_EQ(0, arg->arg("foo"));
}
